//! SeedRL-style centralized inference channels.
//!
//! Actors send observations to a centralized inference server (no GPU needed
//! on actors). Use [`inference_channel`], [`serve_inference_batch`], and the
//! trajectory queue from [`super::trajectory`].

use crate::collect::actor_learner::batched_inference;
use crate::nn::policy::DiscreteActorCritic;

use burn::prelude::*;
use rand::Rng;
use std::sync::mpsc;

/// Request from an actor to the centralized inference server.
pub struct InferenceRequest {
    /// Single observation vector.
    pub observation: Vec<f32>,
    /// One-shot channel for the server's response.
    pub response_tx: mpsc::Sender<InferenceResponse>,
}

/// Response from the inference server to an actor.
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    /// Sampled action index.
    pub action: i32,
    /// Log probability under the behavior policy.
    pub log_prob: f32,
}

/// Actor-side handle for requesting centralized inference.
///
/// Clone this to give each actor thread its own handle.
#[derive(Clone)]
pub struct InferenceHandle {
    tx: mpsc::SyncSender<InferenceRequest>,
}

impl InferenceHandle {
    /// Send an observation and block until the inference server returns an action.
    ///
    /// Returns `None` if the inference server has shut down.
    pub fn infer(&self, observation: Vec<f32>) -> Option<InferenceResponse> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx
            .send(InferenceRequest {
                observation,
                response_tx: resp_tx,
            })
            .ok()?;
        resp_rx.recv().ok()
    }
}

/// Server-side receiver for batching inference requests.
pub struct InferenceReceiver {
    rx: mpsc::Receiver<InferenceRequest>,
}

impl InferenceReceiver {
    /// Collect up to `max_batch` requests. Blocks on the first, then drains.
    ///
    /// Returns an empty Vec if all senders have disconnected.
    pub fn recv_batch(&self, max_batch: usize) -> Vec<InferenceRequest> {
        let mut batch = Vec::with_capacity(max_batch);
        match self.rx.recv() {
            Ok(req) => batch.push(req),
            Err(_) => return batch,
        }
        while batch.len() < max_batch {
            match self.rx.try_recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }
        }
        batch
    }
}

/// Create a channel pair for centralized inference (SeedRL pattern).
///
/// `capacity` controls how many inference requests can be buffered before
/// actors block. A good default is the number of actor threads.
pub fn inference_channel(capacity: usize) -> (InferenceHandle, InferenceReceiver) {
    let (tx, rx) = mpsc::sync_channel(capacity);
    (InferenceHandle { tx }, InferenceReceiver { rx })
}

/// Run batched inference on collected requests and send responses.
///
/// Call this in the inference server loop after [`InferenceReceiver::recv_batch`].
pub fn serve_inference_batch<B, M>(
    model: &M,
    requests: Vec<InferenceRequest>,
    device: &B::Device,
    rng: &mut impl Rng,
) where
    B: Backend,
    M: DiscreteActorCritic<B>,
{
    if requests.is_empty() {
        return;
    }

    let observations: Vec<Vec<f32>> = requests.iter().map(|r| r.observation.clone()).collect();
    let results = batched_inference::<B, M>(model, &observations, device, rng);

    for (req, (action, log_prob)) in requests.into_iter().zip(results) {
        let _ = req.response_tx.send(InferenceResponse { action, log_prob });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_channel_roundtrip() {
        let (handle, receiver) = inference_channel(4);

        let handle_clone = handle.clone();
        let actor = std::thread::spawn(move || handle_clone.infer(vec![1.0, 2.0, 3.0]));

        let requests = receiver.recv_batch(10);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].observation, vec![1.0, 2.0, 3.0]);

        for req in requests {
            req.response_tx
                .send(InferenceResponse {
                    action: 1,
                    log_prob: -0.7,
                })
                .unwrap();
        }

        let response = actor.join().unwrap().unwrap();
        assert_eq!(response.action, 1);
        assert!((response.log_prob - (-0.7)).abs() < 1e-6);
    }

    #[test]
    fn inference_channel_multiple_actors() {
        let (handle, receiver) = inference_channel(8);

        let actors: Vec<_> = (0..4)
            .map(|i| {
                let h = handle.clone();
                std::thread::spawn(move || h.infer(vec![i as f32]))
            })
            .collect();

        let mut all_requests = Vec::new();
        while all_requests.len() < 4 {
            let batch = receiver.recv_batch(4);
            all_requests.extend(batch);
        }
        assert_eq!(all_requests.len(), 4);

        for req in all_requests {
            let action = req.observation[0] as i32;
            req.response_tx
                .send(InferenceResponse {
                    action,
                    log_prob: -1.0,
                })
                .unwrap();
        }

        for actor in actors {
            let response = actor.join().unwrap().unwrap();
            assert!((response.log_prob - (-1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn inference_handle_returns_none_on_disconnect() {
        let (handle, receiver) = inference_channel(1);
        drop(receiver);
        assert!(handle.infer(vec![1.0]).is_none());
    }
}

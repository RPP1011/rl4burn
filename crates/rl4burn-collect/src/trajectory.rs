//! Shared trajectory type and bounded queue for actor→learner communication.

use std::sync::mpsc;

/// A fixed-length trajectory produced by an actor.
///
/// Contains `T` steps of experience plus a bootstrap observation at index `T`.
/// `observations` has length `T + 1`; all other fields have length `T`.
#[derive(Clone)]
pub struct Trajectory {
    /// Observations. Length: `T + 1` (last is bootstrap for V-trace).
    pub observations: Vec<Vec<f32>>,
    /// Discrete actions taken (as indices). Length: `T`.
    pub actions: Vec<i32>,
    /// Rewards received. Length: `T`.
    pub rewards: Vec<f32>,
    /// Whether the episode ended at each step. Length: `T`.
    pub dones: Vec<bool>,
    /// Log probabilities under the behavior policy. Length: `T`.
    pub behavior_log_probs: Vec<f32>,
}

/// Actor-side handle for sending completed trajectories to the learner.
///
/// Clone this to give each actor thread its own handle.
#[derive(Clone)]
pub struct TrajectoryProducer {
    tx: mpsc::SyncSender<Trajectory>,
}

impl TrajectoryProducer {
    /// Send a completed trajectory. Blocks if the queue is full.
    pub fn send(&self, trajectory: Trajectory) -> Result<(), mpsc::SendError<Trajectory>> {
        self.tx.send(trajectory)
    }
}

/// Learner-side handle for receiving trajectory batches.
pub struct TrajectoryConsumer {
    rx: mpsc::Receiver<Trajectory>,
}

impl TrajectoryConsumer {
    /// Receive up to `batch_size` trajectories. Blocks on the first.
    ///
    /// Returns an empty Vec if all producers have disconnected.
    pub fn recv_batch(&self, batch_size: usize) -> Vec<Trajectory> {
        let mut batch = Vec::with_capacity(batch_size);
        match self.rx.recv() {
            Ok(traj) => batch.push(traj),
            Err(_) => return batch,
        }
        while batch.len() < batch_size {
            match self.rx.try_recv() {
                Ok(traj) => batch.push(traj),
                Err(_) => break,
            }
        }
        batch
    }
}

/// Create a bounded channel for passing trajectories from actors to the learner.
///
/// `capacity` controls how many completed trajectories can be buffered.
/// When full, actor threads block (providing natural backpressure).
pub fn trajectory_queue(capacity: usize) -> (TrajectoryProducer, TrajectoryConsumer) {
    let (tx, rx) = mpsc::sync_channel(capacity);
    (TrajectoryProducer { tx }, TrajectoryConsumer { rx })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_lengths_consistent() {
        let traj = Trajectory {
            observations: vec![vec![0.0; 4]; 6],
            actions: vec![0; 5],
            rewards: vec![1.0; 5],
            dones: vec![false; 5],
            behavior_log_probs: vec![-0.5; 5],
        };
        assert_eq!(traj.observations.len(), traj.actions.len() + 1);
        assert_eq!(traj.actions.len(), traj.rewards.len());
        assert_eq!(traj.actions.len(), traj.dones.len());
        assert_eq!(traj.actions.len(), traj.behavior_log_probs.len());
    }

    #[test]
    fn trajectory_queue_send_recv() {
        let (tx, rx) = trajectory_queue(4);
        for i in 0..3 {
            tx.send(Trajectory {
                observations: vec![vec![i as f32]; 3],
                actions: vec![0; 2],
                rewards: vec![1.0; 2],
                dones: vec![false; 2],
                behavior_log_probs: vec![-0.5; 2],
            })
            .unwrap();
        }
        let batch = rx.recv_batch(10);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].observations[0], vec![0.0]);
        assert_eq!(batch[2].observations[0], vec![2.0]);
    }

    #[test]
    fn trajectory_queue_backpressure() {
        let (tx, rx) = trajectory_queue(2);
        let traj = || Trajectory {
            observations: vec![vec![0.0]; 2],
            actions: vec![0],
            rewards: vec![0.0],
            dones: vec![false],
            behavior_log_probs: vec![0.0],
        };
        tx.send(traj()).unwrap();
        tx.send(traj()).unwrap();
        assert!(tx.tx.try_send(traj()).is_err());
        let batch = rx.recv_batch(10);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn consumer_returns_empty_on_disconnect() {
        let (tx, rx) = trajectory_queue(4);
        drop(tx);
        let batch = rx.recv_batch(10);
        assert!(batch.is_empty());
    }
}

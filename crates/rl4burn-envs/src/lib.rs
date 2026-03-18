//! Built-in environments for rl4burn: CartPole, Pendulum, GridWorld.

pub mod cartpole;
pub mod gridworld;
pub mod pendulum;

pub use cartpole::CartPole;
pub use gridworld::GridWorld;
pub use pendulum::Pendulum;

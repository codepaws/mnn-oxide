//! Async-compatible helpers for thread-confined MNN inference.
//!
//! These futures execute MNN calls synchronously on the polling thread. They
//! do not make native handles `Send` or `Sync`, and long inference calls can
//! block an async executor. Applications needing non-blocking inference
//! should own a dedicated worker thread and send it owned input/output data.

#[cfg(feature = "async")]
mod inner {
    use crate::config::ScheduleConfig;
    use crate::error::MnnResult;
    use crate::interpreter::AsyncInterpreter;
    use crate::session::Session;

    /// Run inference using the thread-confined session on the polling thread.
    pub async fn run_session_async(session: &mut Session) -> MnnResult<()> {
        session.run_async().await
    }

    /// A sequential batch helper for local async code.
    ///
    /// Because MNN handles cannot safely move between threads, this helper
    /// processes inputs serially. `max_concurrent` is retained for source
    /// compatibility and is currently ignored. The async methods also execute
    /// native operations synchronously and may block the polling thread.
    #[derive(Debug)]
    pub struct AsyncBatchInference {
        interpreter: AsyncInterpreter,
        config: ScheduleConfig,
    }

    impl AsyncBatchInference {
        /// Create a sequential batch context.
        pub fn new(
            interpreter: AsyncInterpreter,
            config: ScheduleConfig,
            _max_concurrent: usize,
        ) -> Self {
            Self {
                interpreter,
                config,
            }
        }

        /// Process each input serially on the current async task.
        pub async fn run_batch<F, T, R>(
            &self,
            inputs: impl IntoIterator<Item = T>,
            mut process: F,
        ) -> MnnResult<Vec<MnnResult<R>>>
        where
            F: FnMut(&mut Session, T) -> MnnResult<R>,
        {
            let mut results = Vec::new();
            for input in inputs {
                let result = match self.interpreter.create_session(self.config.clone()).await {
                    Ok(mut session) => process(&mut session, input),
                    Err(error) => Err(error),
                };
                results.push(result);
            }
            Ok(results)
        }
    }

    /// A local pool of sessions for sequential inference.
    ///
    /// Session handles are thread-confined; this pool does not provide
    /// cross-thread concurrency.
    #[derive(Debug)]
    pub struct SessionPool {
        interpreter: AsyncInterpreter,
        config: ScheduleConfig,
        pool: Vec<Session>,
    }

    impl SessionPool {
        /// Create a pool containing `pool_size` native sessions.
        pub async fn new(
            interpreter: AsyncInterpreter,
            config: ScheduleConfig,
            pool_size: usize,
        ) -> MnnResult<Self> {
            let mut pool = Vec::with_capacity(pool_size);
            for _ in 0..pool_size {
                pool.push(interpreter.create_session(config.clone()).await?);
            }
            Ok(Self {
                interpreter,
                config,
                pool,
            })
        }

        /// Acquire one session from this thread-local pool.
        pub fn acquire(&mut self) -> Option<Session> {
            self.pool.pop()
        }

        /// Return a session to this thread-local pool.
        pub fn release(&mut self, session: Session) {
            self.pool.push(session);
        }

        /// Return the number of currently available sessions.
        pub fn available(&self) -> usize {
            self.pool.len()
        }

        /// Create another session from the pool's configuration.
        pub async fn create_session(&self) -> MnnResult<Session> {
            self.interpreter.create_session(self.config.clone()).await
        }
    }
}

#[cfg(feature = "async")]
pub use inner::*;

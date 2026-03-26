use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone)]
pub enum TaskStatus {
    Running,
    Done(String),
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct BackgroundTask {
    pub id: u32,
    pub name: String,
    pub command: String,
    pub status: TaskStatus,
    pub started: Instant,
    pub elapsed_ms: Option<u64>,
}

/// Shared task table for background tool executions.
#[derive(Clone)]
pub struct TaskTable {
    inner: Arc<Mutex<TaskTableInner>>,
}

struct TaskTableInner {
    next_id: u32,
    tasks: HashMap<u32, BackgroundTask>,
    notified: std::collections::HashSet<u32>,
}

impl TaskTable {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TaskTableInner {
                next_id: 1,
                tasks: HashMap::new(),
                notified: std::collections::HashSet::new(),
            })),
        }
    }

    /// Register a new background task. Returns the task ID.
    pub fn register(&self, name: &str, command: &str) -> u32 {
        let mut inner = self.inner.lock().unwrap();
        let id = inner.next_id;
        inner.next_id += 1;
        inner.tasks.insert(
            id,
            BackgroundTask {
                id,
                name: name.to_string(),
                command: command.to_string(),
                status: TaskStatus::Running,
                started: Instant::now(),
                elapsed_ms: None,
            },
        );
        id
    }

    /// Mark a task as completed with output.
    pub fn complete(&self, id: u32, output: String) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(task) = inner.tasks.get_mut(&id) {
            task.elapsed_ms = Some(task.started.elapsed().as_millis() as u64);
            task.status = TaskStatus::Done(output);
        }
    }

    /// Mark a task as failed.
    pub fn fail(&self, id: u32, error: String) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(task) = inner.tasks.get_mut(&id) {
            task.elapsed_ms = Some(task.started.elapsed().as_millis() as u64);
            task.status = TaskStatus::Failed(error);
        }
    }

    /// List all tasks.
    pub fn list(&self) -> Vec<BackgroundTask> {
        let inner = self.inner.lock().unwrap();
        let mut tasks: Vec<_> = inner.tasks.values().cloned().collect();
        tasks.sort_by_key(|t| t.id);
        tasks
    }

    /// Return tasks that completed since the last call to this method.
    pub fn take_new_completions(&self) -> Vec<BackgroundTask> {
        let mut inner = self.inner.lock().unwrap();
        let mut result = Vec::new();
        for (id, task) in &inner.tasks {
            if !matches!(task.status, TaskStatus::Running) && !inner.notified.contains(id) {
                result.push(task.clone());
            }
        }
        for task in &result {
            inner.notified.insert(task.id);
        }
        result
    }

    /// Format task list for display.
    pub fn format_list(&self) -> String {
        let tasks = self.list();
        if tasks.is_empty() {
            return "No background tasks.".to_string();
        }
        let mut lines = vec!["Background tasks:".to_string()];
        for task in &tasks {
            let status = match &task.status {
                TaskStatus::Running => {
                    let elapsed = task.started.elapsed().as_millis();
                    format!("running ({elapsed} ms)")
                }
                TaskStatus::Done(output) => {
                    let preview = if output.len() > 60 {
                        format!("{}...", &output[..60])
                    } else {
                        output.clone()
                    };
                    let ms = task.elapsed_ms.unwrap_or(0);
                    format!("done ({ms} ms): {preview}")
                }
                TaskStatus::Failed(err) => {
                    let ms = task.elapsed_ms.unwrap_or(0);
                    format!("failed ({ms} ms): {err}")
                }
            };
            lines.push(format!(
                "  [{}] {} `{}` — {}",
                task.id, task.name, task.command, status
            ));
        }
        lines.join("\n")
    }
}

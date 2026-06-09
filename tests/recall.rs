use olorin::kernels::ffi;
use olorin::recall::VectorStore;

#[test]
fn test_recall_basic() {
    ffi::init().unwrap();
    let mut store = VectorStore::new(16);
    store.add("The weather is sunny today");
    store.add("Rust programming is fun");
    let results = store.search("weather forecast", 1);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_recall_empty() {
    ffi::init().unwrap();
    let mut store = VectorStore::new(16);
    let results = store.search("anything", 5);
    assert!(results.is_empty());
}

#[test]
fn test_recall_len() {
    ffi::init().unwrap();
    let mut store = VectorStore::new(16);
    assert_eq!(store.len(), 0);
    store.add("first entry");
    assert_eq!(store.len(), 1);
    store.add("second entry");
    assert_eq!(store.len(), 2);
}

#[test]
fn test_recall_ring_buffer() {
    ffi::init().unwrap();
    let mut store = VectorStore::new(2);
    store.add("alpha content");
    store.add("beta content");
    store.add("gamma content"); // evicts alpha
    assert_eq!(store.len(), 2);
    // alpha should be gone
    let results = store.search("alpha", 2);
    for r in &results {
        assert!(!r.text.contains("alpha"), "alpha should be evicted, got: {}", r.text);
    }
}

#[test]
fn test_recall_clear() {
    ffi::init().unwrap();
    let mut store = VectorStore::new(16);
    store.add("something important");
    assert_eq!(store.len(), 1);
    store.clear();
    assert_eq!(store.len(), 0);
    assert!(store.search("something", 1).is_empty());
}

#[test]
fn test_recall_similar_content_ranks_higher() {
    ffi::init().unwrap();
    let mut store = VectorStore::new(32);
    store.add("rust programming language systems");
    store.add("12345 67890 numbers only content");
    let results = store.search("rust systems programming", 1);
    assert_eq!(results.len(), 1);
    assert!(results[0].text.contains("rust"), "rust entry should rank highest");
}

#[test]
fn synthesize_prefers_fresh_fact_over_stale_after_update() {
    // Regression: a fact update ("my name is now Retep") must win over the
    // stale fact ("my name is Peter") when the question is asked again.
    // Previously the query echo ("what is my name") topped the results, dedup
    // absorbed the fresh answer as a near-duplicate of the echo, and then the
    // self-match filter removed the echo — leaving only the stale fact.
    ffi::init().unwrap();
    let mut store = VectorStore::new(1024);
    store.add("my name is Peter");
    store.add("what is my name");
    store.add("my name is now Retep");
    let ctx = store
        .synthesize_context("what is my name?", 1)
        .expect("expected recalled context");
    assert!(ctx.contains("now Retep"), "should recall the updated fact, got:\n{ctx}");
    assert!(!ctx.contains("Peter"), "should not recall the stale fact, got:\n{ctx}");
}

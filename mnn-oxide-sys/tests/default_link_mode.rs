#[test]
fn sys_crate_defaults_to_dynamic_mnn_linking() {
    assert!(cfg!(feature = "dynamic"));
    assert!(!cfg!(feature = "static"));
}

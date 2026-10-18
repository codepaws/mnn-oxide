#[cfg(all(feature = "dynamic", not(feature = "static")))]
#[test]
fn cargo_defaults_to_dynamic_mnn_linking() {
    assert!(cfg!(feature = "dynamic"));
    assert!(!cfg!(feature = "static"));
}

#[cfg(all(feature = "static", not(feature = "dynamic")))]
#[test]
fn explicit_static_linking_selects_static_mode() {
    assert!(cfg!(feature = "static"));
    assert!(!cfg!(feature = "dynamic"));
}

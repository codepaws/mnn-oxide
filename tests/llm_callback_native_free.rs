#[path = "../src/llm_callback.rs"]
mod llm_callback;

use llm_callback::{stream_callback, CallbackState};
use std::cell::Cell;
use std::ffi::{c_void, CString};
use std::rc::Rc;

#[test]
fn callback_panic_is_caught_and_later_chunks_are_suppressed() {
    let calls = Rc::new(Cell::new(0));
    let calls_in_callback = calls.clone();
    let mut state = CallbackState::<Box<dyn FnMut(&str)>> {
        on_chunk: Box::new(move |_| {
            calls_in_callback.set(calls_in_callback.get() + 1);
            panic!("callback failure");
        }),
        panicked: false,
    };
    let text = CString::new("first chunk").unwrap();
    let userdata = (&mut state as *mut CallbackState<_>).cast::<c_void>();

    unsafe { stream_callback::<Box<dyn FnMut(&str)>>(text.as_ptr(), userdata) };
    assert!(state.panicked);
    assert_eq!(calls.get(), 1);

    unsafe { stream_callback::<Box<dyn FnMut(&str)>>(text.as_ptr(), userdata) };
    assert_eq!(calls.get(), 1);
}

#[test]
fn callback_delivers_nul_terminated_text() {
    let received = Rc::new(std::cell::RefCell::new(String::new()));
    let received_in_callback = received.clone();
    let mut state = CallbackState::<Box<dyn FnMut(&str)>> {
        on_chunk: Box::new(move |chunk: &str| received_in_callback.borrow_mut().push_str(chunk)),
        panicked: false,
    };
    let text = CString::new("a chunk").unwrap();
    let userdata = (&mut state as *mut CallbackState<_>).cast::<c_void>();

    unsafe { stream_callback::<Box<dyn FnMut(&str)>>(text.as_ptr(), userdata) };

    assert_eq!(&*received.borrow(), "a chunk");
    assert!(!state.panicked);
}

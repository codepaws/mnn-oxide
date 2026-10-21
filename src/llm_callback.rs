use std::ffi::{c_char, c_void, CStr};

pub(crate) struct CallbackState<F> {
    pub(crate) on_chunk: F,
    pub(crate) panicked: bool,
}

/// Keep all Rust work behind the C ABI boundary inside `catch_unwind`.
pub(crate) unsafe extern "C" fn stream_callback<F>(text: *const c_char, userdata: *mut c_void)
where
    F: FnMut(&str),
{
    if userdata.is_null() {
        return;
    }
    // SAFETY: the caller keeps this state alive for the synchronous native
    // generation call and passes the exact concrete type used here.
    let state = unsafe { &mut *(userdata as *mut CallbackState<F>) };
    if state.panicked {
        return;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if text.is_null() {
            return;
        }
        // SAFETY: LlmStreamBuf passes a NUL-terminated string for this call.
        let chunk = unsafe { CStr::from_ptr(text) }.to_string_lossy();
        (state.on_chunk)(&chunk);
    }));
    if result.is_err() {
        state.panicked = true;
    }
}

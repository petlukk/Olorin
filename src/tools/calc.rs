use super::ToolResult;

pub fn run(args: &str) -> ToolResult {
    let expr = args.trim();
    if expr.is_empty() {
        return ToolResult { output: "usage: calc <expression>".to_string(), success: false };
    }
    match expr_eval(expr) {
        Ok(val) => ToolResult { output: val, success: true },
        Err(e) => ToolResult { output: format!("calc error: {e}"), success: false },
    }
}

const SCALE: i64 = 1_000_000;

fn expr_eval(expr: &str) -> Result<String, String> {
    let mut val_stack: Vec<i64> = vec![0i64; 32];
    let mut op_stack: Vec<i32> = vec![0i32; 32];

    let bytes = expr.as_bytes();
    let mut out_result: i64 = 0;
    let mut out_error: i32 = 0;

    unsafe {
        crate::kernels::ffi::eval_expr(
            bytes.as_ptr(),
            bytes.len() as i32,
            &mut out_result,
            &mut out_error,
            val_stack.as_mut_ptr(),
            op_stack.as_mut_ptr(),
        );
    }

    if out_error != 0 {
        return Err(match out_error {
            1 => "division by zero".to_string(),
            2 => "invalid expression".to_string(),
            3 => "overflow".to_string(),
            _ => format!("error code {out_error}"),
        });
    }

    Ok(format_fixed_point(out_result))
}

fn format_fixed_point(value: i64) -> String {
    let negative = value < 0;
    let abs = value.unsigned_abs();
    let whole = abs / SCALE as u64;
    let frac = abs % SCALE as u64;
    if frac == 0 {
        if negative { format!("-{whole}") } else { format!("{whole}") }
    } else {
        let frac_str = format!("{:06}", frac);
        let trimmed = frac_str.trim_end_matches('0');
        if negative { format!("-{whole}.{trimmed}") } else { format!("{whole}.{trimmed}") }
    }
}

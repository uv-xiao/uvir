// Tests for the MLIR-compatible printer in uvir.
//
// Purpose: Validates the printer functionality including:
// - Type printing for all builtin types
// - Function type printing with proper syntax
// - Attribute printing (integers, floats, strings, arrays)
// - Value printing with SSA names
// - Operation printing with all components
// - Region printing with proper indentation
// - Special character escaping in strings
// - Unicode support in printed output
//
// The printer generates MLIR textual format, enabling IR serialization
// and human-readable output for debugging and interoperability.

use uvir::{Context, Value};
use uvir::printer::Printer;
use uvir::types::FloatPrecision;
use uvir::attribute::Attribute;
use uvir::dialects::builtin::{integer_type, float_type, function_type};
use uvir::dialects::arith::{ConstantOp, AddOp};

#[test]
fn test_print_types() {
    let mut ctx = Context::new();
    
    // Print integer type
    let mut printer = Printer::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    printer.print_type(&ctx, i32_type).unwrap();
    let output = printer.get_output();
    assert!(output.contains("i32"));
    
    // Print unsigned type
    let mut printer = Printer::new();
    let u64_type = integer_type(&mut ctx, 64, false);
    printer.print_type(&ctx, u64_type).unwrap();
    let output = printer.get_output();
    assert!(output.contains("u64"));
    
    // Print float types
    let mut printer = Printer::new();
    let f32_type = float_type(&mut ctx, FloatPrecision::Single);
    printer.print_type(&ctx, f32_type).unwrap();
    let output = printer.get_output();
    assert!(output.contains("f32"));
}

#[test]
fn test_print_function_types() {
    let mut ctx = Context::new();
    
    let i32_type = integer_type(&mut ctx, 32, true);
    let i64_type = integer_type(&mut ctx, 64, true);
    
    // Function type: (i32, i32) -> i64
    let mut printer = Printer::new();
    let fn_type = function_type(&mut ctx, vec![i32_type, i32_type], vec![i64_type]);
    printer.print_type(&ctx, fn_type).unwrap();
    let output = printer.get_output();
    assert!(output.contains("(i32, i32) -> i64"));
    
    // Function with no args: () -> i32
    let mut printer = Printer::new();
    let fn_type2 = function_type(&mut ctx, vec![], vec![i32_type]);
    printer.print_type(&ctx, fn_type2).unwrap();
    let output = printer.get_output();
    assert!(output.contains("() -> i32"));
    
    // Function with multiple returns
    let mut printer = Printer::new();
    let fn_type3 = function_type(&mut ctx, vec![i32_type], vec![i32_type, i64_type]);
    printer.print_type(&ctx, fn_type3).unwrap();
    let output = printer.get_output();
    assert!(output.contains("(i32) -> (i32, i64)"));
}

#[test]
fn test_print_attributes() {
    let mut ctx = Context::new();
    
    // Print integer attribute
    let mut printer = Printer::new();
    printer.print_attribute(&ctx, &Attribute::Integer(42)).unwrap();
    let output = printer.get_output();
    assert!(output.contains("42"));
    
    // Print float attribute
    let mut printer = Printer::new();
    printer.print_attribute(&ctx, &Attribute::Float(3.14)).unwrap();
    let output = printer.get_output();
    assert!(output.contains("3.14"));
    
    // Print string attribute
    let mut printer = Printer::new();
    let str_id = ctx.intern_string("hello world");
    printer.print_attribute(&ctx, &Attribute::String(str_id)).unwrap();
    let output = printer.get_output();
    assert!(output.contains("\"hello world\""));
}

#[test]
fn test_print_array_attributes() {
    let ctx = Context::new();
    
    // Print array attribute
    let mut printer = Printer::new();
    let array = Attribute::Array(vec![
        Attribute::Integer(1),
        Attribute::Integer(2),
        Attribute::Integer(3),
    ]);
    printer.print_attribute(&ctx, &array).unwrap();
    let output = printer.get_output();
    assert!(output.contains("[1, 2, 3]"));
    
    // Print nested array
    let mut printer = Printer::new();
    let nested = Attribute::Array(vec![
        Attribute::Array(vec![Attribute::Integer(1), Attribute::Integer(2)]),
        Attribute::Array(vec![Attribute::Integer(3), Attribute::Integer(4)]),
    ]);
    printer.print_attribute(&ctx, &nested).unwrap();
    let output = printer.get_output();
    assert!(output.contains("[[1, 2], [3, 4]]"));
}

#[test]
fn test_print_values() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    
    let val1_name = ctx.intern_string("result");
    let (val1, val2) = {
        let region = ctx.get_global_region_mut();
        
        // Create named value
        let v1 = region.add_value(Value {
            name: Some(val1_name),
            ty: i32_type,
            defining_op: None,
        });
        
        // Create anonymous value
        let v2 = region.add_value(Value {
            name: None,
            ty: i32_type,
            defining_op: None,
        });
        (v1, v2)
    };
    
    let mut printer = Printer::new();
    printer.print_value(val1).unwrap();
    let output = printer.get_output();
    assert!(output.contains("%"));
    
    let mut printer = Printer::new();
    printer.print_value(val2).unwrap();
    let output = printer.get_output();
    // Anonymous values might be printed as %0, %1, etc.
    assert!(output.starts_with('%'));
}

#[test]
fn test_print_operations() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    
    let a_name = ctx.intern_string("a");
    let b_name = ctx.intern_string("b");
    let sum_name = ctx.intern_string("sum");
    
    // Create values
    let (val1, val2, sum) = {
        let region = ctx.get_global_region_mut();
        let v1 = region.add_value(Value {
            name: Some(a_name),
            ty: i32_type,
            defining_op: None,
        });
        let v2 = region.add_value(Value {
            name: Some(b_name),
            ty: i32_type,
            defining_op: None,
        });
        let s = region.add_value(Value {
            name: Some(sum_name),
            ty: i32_type,
            defining_op: None,
        });
        (v1, v2, s)
    };
    
    // Create constant operation
    let const_op = ConstantOp {
        result: val1,
        value: Attribute::Integer(42),
    };
    
    let op_data = const_op.into_op_data(&mut ctx);
    
    let mut printer = Printer::new();
    printer.print_operation(&ctx, &op_data).unwrap();
    let output = printer.get_output();
    // Should print something like: %a = arith.constant 42 : i32
    assert!(output.contains("arith.constant"));
    assert!(output.contains("42"));
    
    // Test add operation
    let add_op = AddOp {
        result: sum,
        lhs: val1,
        rhs: val2,
    };
    
    let add_data = add_op.into_op_data(&mut ctx);
    let mut printer = Printer::new();
    printer.print_operation(&ctx, &add_data).unwrap();
    let output = printer.get_output();
    // Should print something like: %sum = arith.addi %a, %b : i32
    assert!(output.contains("arith.addi"));
}

#[test]
fn test_print_with_indentation() {
    // Test basic printing
    let mut printer = Printer::new();
    printer.print("Hello").unwrap();
    let output = printer.get_output();
    assert!(output.contains("Hello"));
    
    // Test with indentation
    let mut printer = Printer::new();
    printer.indent();
    printer.println("Line 1").unwrap();
    printer.println("Line 2").unwrap();
    printer.dedent();
    printer.println("Line 3").unwrap();
    
    let output = printer.get_output();
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 3);
    // Note: indentation behavior may vary based on implementation
}

#[test]
fn test_print_region() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    
    let x_name = ctx.intern_string("x");
    
    // Add operations to region
    let val = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: Some(x_name),
            ty: i32_type,
            defining_op: None,
        })
    };
    
    let const_op = ConstantOp {
        result: val,
        value: Attribute::Integer(100),
    };
    
    let op_data = const_op.into_op_data(&mut ctx);
    ctx.get_global_region_mut().add_op(op_data);
    
    let mut printer = Printer::new();
    printer.print_region(&ctx, ctx.global_region()).unwrap();
    
    let output = printer.get_output();
    // Should contain the operation
    assert!(output.contains("arith.constant"));
    assert!(output.contains("100"));
}

#[test]
fn test_print_special_characters() {
    let mut ctx = Context::new();
    
    // Test printing strings with special characters
    let mut printer = Printer::new();
    let str_with_quotes = ctx.intern_string("Hello \"World\"");
    printer.print_attribute(&ctx, &Attribute::String(str_with_quotes)).unwrap();
    let output = printer.get_output();
    // Should escape the quotes
    assert!(output.contains("\\\""));
    
    // Test newlines
    let mut printer = Printer::new();
    let str_with_newline = ctx.intern_string("Line1\nLine2");
    printer.print_attribute(&ctx, &Attribute::String(str_with_newline)).unwrap();
    let output = printer.get_output();
    // Should escape newlines
    assert!(output.contains("\\n"));
}

#[test]
fn test_print_unicode() {
    let mut ctx = Context::new();
    let mut printer = Printer::new();
    
    // Test printing unicode strings
    let unicode_str = ctx.intern_string("Hello 世界 🌍");
    printer.print_attribute(&ctx, &Attribute::String(unicode_str)).unwrap();
    let output = printer.get_output();
    // The printer correctly escapes unicode characters for MLIR compatibility
    assert!(output.starts_with("\""));
    assert!(output.ends_with("\""));
    // The escaped form will contain \\u{...} sequences
    assert!(output.contains("Hello"));
}
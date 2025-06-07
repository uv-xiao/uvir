// Tests for the type system and type interning in uvir.
//
// Purpose: Validates the type system implementation including:
// - Type interning for memory efficiency and fast equality checks
// - Builtin types (integers, floats, functions)
// - Type deduplication across the context
// - Custom dialect types through the type erasure mechanism
// - Complex nested types (e.g., functions returning functions)
//
// The type system is fundamental to IR correctness as it ensures
// type safety and enables type-based optimizations.

use uvir::dialects::builtin::{float_type, function_type, integer_type};
use uvir::parser::Parser;
use uvir::printer::Printer;
use uvir::types::{DialectType, TypeStorage};
use uvir::{Context, FloatPrecision, TypeKind};

#[test]
fn test_builtin_integer_types() {
    let mut ctx = Context::new();

    // Test various integer types
    let i8 = integer_type(&mut ctx, 8, true);
    let i16 = integer_type(&mut ctx, 16, true);
    let i32 = integer_type(&mut ctx, 32, true);
    let i64 = integer_type(&mut ctx, 64, true);

    let u8 = integer_type(&mut ctx, 8, false);
    let u16 = integer_type(&mut ctx, 16, false);
    let u32 = integer_type(&mut ctx, 32, false);
    let _u64 = integer_type(&mut ctx, 64, false);

    // Different types should have different IDs
    assert_ne!(i8, i16);
    assert_ne!(i32, i64);
    assert_ne!(i32, u32);

    // Test type interning - same type should get same ID
    let i32_2 = integer_type(&mut ctx, 32, true);
    let u32_2 = integer_type(&mut ctx, 32, false);

    assert_eq!(i32, i32_2);
    assert_eq!(u32, u32_2);
}

#[test]
fn test_builtin_float_types() {
    let mut ctx = Context::new();

    let f16 = float_type(&mut ctx, FloatPrecision::Half);
    let f32 = float_type(&mut ctx, FloatPrecision::Single);
    let f64 = float_type(&mut ctx, FloatPrecision::Double);

    // Different precisions should have different IDs
    assert_ne!(f16, f32);
    assert_ne!(f32, f64);
    assert_ne!(f16, f64);

    // Test interning
    let f32_2 = float_type(&mut ctx, FloatPrecision::Single);
    assert_eq!(f32, f32_2);
}

#[test]
fn test_function_types() {
    let mut ctx = Context::new();

    let i32 = integer_type(&mut ctx, 32, true);
    let i64 = integer_type(&mut ctx, 64, true);
    let f32 = float_type(&mut ctx, FloatPrecision::Single);

    // Function with no args returning i32
    let fn1 = function_type(&mut ctx, vec![], vec![i32]);

    // Function taking i32, returning i64
    let fn2 = function_type(&mut ctx, vec![i32], vec![i64]);

    // Function taking (i32, i64) returning f32
    let fn3 = function_type(&mut ctx, vec![i32, i64], vec![f32]);

    // Function with multiple returns
    let fn4 = function_type(&mut ctx, vec![i32], vec![i32, i64]);

    // All should be different
    assert_ne!(fn1, fn2);
    assert_ne!(fn2, fn3);
    assert_ne!(fn3, fn4);

    // Test interning
    let fn2_dup = function_type(&mut ctx, vec![i32], vec![i64]);
    assert_eq!(fn2, fn2_dup);
}

#[test]
fn test_type_retrieval() {
    let mut ctx = Context::new();

    let i32 = integer_type(&mut ctx, 32, true);
    let f64 = float_type(&mut ctx, FloatPrecision::Double);
    let fn_type = function_type(&mut ctx, vec![i32, i32], vec![f64]);

    // Should be able to retrieve type information
    match ctx.get_type(i32) {
        Some(TypeKind::Integer { width, signed }) => {
            assert_eq!(*width, 32);
            assert_eq!(*signed, true);
        }
        _ => panic!("Expected integer type"),
    }

    match ctx.get_type(f64) {
        Some(TypeKind::Float { precision }) => {
            assert_eq!(*precision, FloatPrecision::Double);
        }
        _ => panic!("Expected float type"),
    }

    match ctx.get_type(fn_type) {
        Some(TypeKind::Function { inputs, outputs }) => {
            assert_eq!(inputs.len(), 2);
            assert_eq!(outputs.len(), 1);
            assert_eq!(inputs[0], i32);
            assert_eq!(inputs[1], i32);
            assert_eq!(outputs[0], f64);
        }
        _ => panic!("Expected function type"),
    }
}

#[test]
fn test_type_id_properties() {
    let mut ctx = Context::new();

    let i32 = integer_type(&mut ctx, 32, true);
    let i64 = integer_type(&mut ctx, 64, true);

    // TypeId should be copyable
    let i32_copy = i32;
    assert_eq!(i32, i32_copy);

    // TypeId should be hashable
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert(i32, "thirty-two");
    map.insert(i64, "sixty-four");

    assert_eq!(map.get(&i32), Some(&"thirty-two"));
    assert_eq!(map.get(&i64), Some(&"sixty-four"));
}

// Define a custom type
#[derive(Clone, PartialEq, Debug)]
struct CustomType {
    size: u32,
    name: String,
}

impl CustomType {
    fn parse(_parser: &mut Parser) -> uvir::Result<Self> {
        Ok(CustomType {
            size: 42,
            name: "custom".to_string(),
        })
    }

    fn print(&self, printer: &mut Printer) -> uvir::Result<()> {
        printer.print(&format!("custom<{}, {}>", self.size, self.name))?;
        Ok(())
    }
}

uvir::impl_dialect_type!(CustomType);

#[test]
fn test_custom_dialect_type() {

    let mut ctx = Context::new();
    let dialect_name = ctx.intern_string("test_dialect");

    // Create custom type storage
    let custom = CustomType {
        size: 100,
        name: "test".to_string(),
    };
    let storage = TypeStorage::new(custom);


    let type_kind = TypeKind::Dialect {
        dialect: dialect_name,
        data: storage,
    };

    // Intern the custom type
    let type_id = ctx.intern_type(type_kind);

    // Should be able to retrieve it
    match ctx.get_type(type_id) {
        Some(TypeKind::Dialect { dialect, data }) => {
            assert_eq!(*dialect, dialect_name);
            // We can't easily test the data without exposing internals
            assert!(data.as_ref::<CustomType>().is_some());
        }
        _ => panic!("Expected dialect type"),
    }
}

#[test]
fn test_complex_nested_types() {
    let mut ctx = Context::new();

    let i32 = integer_type(&mut ctx, 32, true);
    let i64 = integer_type(&mut ctx, 64, true);
    let f32 = float_type(&mut ctx, FloatPrecision::Single);

    // Function returning another function
    let inner_fn = function_type(&mut ctx, vec![i32], vec![i64]);
    let outer_fn = function_type(&mut ctx, vec![f32], vec![inner_fn]);

    // Function taking and returning multiple values
    let complex_fn = function_type(&mut ctx, vec![i32, i64, f32], vec![i64, i32]);

    assert_ne!(inner_fn, outer_fn);
    assert_ne!(outer_fn, complex_fn);

    // Verify structure
    match ctx.get_type(outer_fn) {
        Some(TypeKind::Function { inputs, outputs }) => {
            assert_eq!(inputs.len(), 1);
            assert_eq!(outputs.len(), 1);
            assert_eq!(inputs[0], f32);

            // The output should be a function type
            match ctx.get_type(outputs[0]) {
                Some(TypeKind::Function {
                    inputs: inner_inputs,
                    outputs: inner_outputs,
                }) => {
                    assert_eq!(inner_inputs.len(), 1);
                    assert_eq!(inner_outputs.len(), 1);
                    assert_eq!(inner_inputs[0], i32);
                    assert_eq!(inner_outputs[0], i64);
                }
                _ => panic!("Expected inner function type"),
            }
        }
        _ => panic!("Expected outer function type"),
    }
}

#[test]
fn test_many_types_interning() {
    let mut ctx = Context::new();
    let mut type_ids = Vec::new();

    // Create many different integer types
    for width in [8, 16, 32, 64, 128].iter() {
        for &signed in &[true, false] {
            let ty = integer_type(&mut ctx, *width, signed);
            type_ids.push((ty, *width, signed));
        }
    }

    // Verify all are different
    for i in 0..type_ids.len() {
        for j in i + 1..type_ids.len() {
            assert_ne!(
                type_ids[i].0, type_ids[j].0,
                "Types {:?} and {:?} should be different",
                type_ids[i], type_ids[j]
            );
        }
    }

    // Re-intern and verify we get the same IDs
    for (expected_id, width, signed) in &type_ids {
        let ty = integer_type(&mut ctx, *width, *signed);
        assert_eq!(ty, *expected_id);
    }
}

#[test]
fn test_type_equality() {
    let mut ctx = Context::new();

    // Create some types
    let i32_1 = integer_type(&mut ctx, 32, true);
    let i32_2 = integer_type(&mut ctx, 32, true);
    let u32 = integer_type(&mut ctx, 32, false);
    let i64 = integer_type(&mut ctx, 64, true);

    // Same types should be equal
    assert_eq!(i32_1, i32_2);

    // Different types should not be equal
    assert_ne!(i32_1, u32); // signed vs unsigned
    assert_ne!(i32_1, i64); // different width
    assert_ne!(u32, i64); // both different
}

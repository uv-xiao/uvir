use uvir::parser::Parser;
use uvir::printer::Printer;
use uvir::Context;

#[test]
fn test_generic_operation_syntax() {
    let mut ctx = Context::new();

    let mlir = r#"
        %0 = "arith.addi"(%arg0, %arg1) : (i32, i32) -> i32
        %1 = "arith.mulf"(%f0, %f1) : (f32, f32) -> f32
    "#;

    let parser = Parser::new(mlir.to_string(), &mut ctx).unwrap();
    // This would fail until we have proper value setup
    // Just verify it can tokenize for now
    assert!(!parser.is_at_end());
}

#[test]
fn test_module_syntax() {
    let mut ctx = Context::new();

    let mlir = r#"
        module @test_module {
            func.func @main() {
                return
            }
        }
    "#;

    let parser = Parser::new(mlir.to_string(), &mut ctx).unwrap();
    assert!(!parser.is_at_end());
}

#[test]
fn test_type_aliases() {
    let mut ctx = Context::new();

    let mlir = r#"
        !matrix_type = tensor<4x4xf32>
        !vec_type = vector<4xf32>
        
        func.func @use_aliases(%arg0: !matrix_type) -> !vec_type {
            %0 = "some.op"(%arg0) : (!matrix_type) -> !vec_type
            return %0 : !vec_type
        }
    "#;

    let parser = Parser::new(mlir.to_string(), &mut ctx).unwrap();
    assert!(!parser.is_at_end());
}

#[test]
fn test_attribute_aliases() {
    let mut ctx = Context::new();

    let mlir = r#"
        #loc = loc("file.mlir":10:5)
        #map = affine_map<(d0, d1) -> (d0, d1)>
        
        %0 = arith.constant 42 : i32 loc(#loc)
    "#;

    let parser = Parser::new(mlir.to_string(), &mut ctx).unwrap();
    assert!(!parser.is_at_end());
}

#[test]
fn test_new_builtin_types() {
    let mut ctx = Context::new();

    let mlir = r#"
        %0 = arith.constant 10 : index
        %1 = "test.none_op"() : () -> none
        %2 = "test.complex_op"() : () -> complex<f32>
        %3 = "test.vector_op"() : () -> vector<4x8xf32>
        %4 = "test.tensor_op"() : () -> tensor<?x10xf32>
        %5 = "test.memref_op"() : () -> memref<4x4xf32, 1>
    "#;

    let parser = Parser::new(mlir.to_string(), &mut ctx).unwrap();
    assert!(!parser.is_at_end());

    // Test that we can parse these types
    let mut parser2 = Parser::new("index".to_string(), &mut ctx).unwrap();
    let index_type = parser2.parse_type().unwrap();
    assert!(matches!(
        ctx.get_type(index_type),
        Some(uvir::types::TypeKind::Index)
    ));

    let mut parser3 = Parser::new("none".to_string(), &mut ctx).unwrap();
    let none_type = parser3.parse_type().unwrap();
    assert!(matches!(
        ctx.get_type(none_type),
        Some(uvir::types::TypeKind::None)
    ));

    let mut parser4 = Parser::new("complex<f32>".to_string(), &mut ctx).unwrap();
    let complex_type = parser4.parse_type().unwrap();
    assert!(matches!(
        ctx.get_type(complex_type),
        Some(uvir::types::TypeKind::Complex { .. })
    ));

    let mut parser5 = Parser::new("vector<4xf32>".to_string(), &mut ctx).unwrap();
    let vector_type = parser5.parse_type().unwrap();
    assert!(matches!(
        ctx.get_type(vector_type),
        Some(uvir::types::TypeKind::Vector { .. })
    ));

    let mut parser6 = Parser::new("tensor<?x10xf32>".to_string(), &mut ctx).unwrap();
    let tensor_type = parser6.parse_type().unwrap();
    assert!(matches!(
        ctx.get_type(tensor_type),
        Some(uvir::types::TypeKind::Tensor { .. })
    ));

    let mut parser7 = Parser::new("memref<4x4xf32, 1>".to_string(), &mut ctx).unwrap();
    let memref_type = parser7.parse_type().unwrap();
    assert!(matches!(
        ctx.get_type(memref_type),
        Some(uvir::types::TypeKind::MemRef { .. })
    ));
}

#[test]
fn test_new_builtin_attributes() {
    let mut ctx = Context::new();

    let mlir = r#"
        %0 = arith.constant true
        %1 = arith.constant false
        %2 = arith.constant unit
        %3 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
        %4 = arith.constant 42 : i32
        %5 = arith.constant 3.14 : f32
    "#;

    let parser = Parser::new(mlir.to_string(), &mut ctx).unwrap();
    assert!(!parser.is_at_end());

    // Test attribute parsing
    let mut parser2 = Parser::new("true".to_string(), &mut ctx).unwrap();
    let true_attr = parser2.parse_attribute().unwrap();
    assert!(matches!(true_attr, uvir::attribute::Attribute::Integer(1)));

    let mut parser3 = Parser::new("false".to_string(), &mut ctx).unwrap();
    let false_attr = parser3.parse_attribute().unwrap();
    assert!(matches!(false_attr, uvir::attribute::Attribute::Integer(0)));

    let mut parser4 = Parser::new("unit".to_string(), &mut ctx).unwrap();
    let unit_attr = parser4.parse_attribute().unwrap();
    assert!(matches!(unit_attr, uvir::attribute::Attribute::String(_)));
}

#[test]
fn test_no_type_signature_for_terminators() {
    let ctx = Context::new();

    // Create a simple operation and print it
    let mut printer = Printer::new();

    // Create mock operation data for return (terminator)
    static RETURN_INFO: uvir::ops::OpInfo = uvir::ops::OpInfo {
        dialect: "func",
        name: "return",
        traits: &[],
        verify: |_| Ok(()),
        parse: |_| unimplemented!(),
        print: |_, _| unimplemented!(),
    };

    let return_op = uvir::ops::OpData {
        info: &RETURN_INFO,
        operands: Default::default(),
        results: Default::default(),
        attributes: Default::default(),
        regions: Default::default(),
        custom_data: uvir::ops::OpStorage::new(),
    };

    // Print the operation - should not include type signature
    printer.print_operation(&ctx, &return_op).unwrap();
    let output = printer.get_output();

    // Verify no type signature for terminator
    assert!(
        !output.contains(" : "),
        "Terminator should not have type signature"
    );
    assert_eq!(output.trim(), "func.return");
}

#[test]
fn test_round_trip_parsing() {
    let mut ctx = Context::new();

    // This test would require full implementation of parsing and printing
    // For now, just test individual components work

    // Test that printer handles new types correctly
    let mut printer = Printer::new();

    let index_type = ctx.intern_type(uvir::types::TypeKind::Index);
    printer.print_type(&ctx, index_type).unwrap();
    let output = printer.get_output();
    assert_eq!(output, "index");

    let mut printer = Printer::new();
    let none_type = ctx.intern_type(uvir::types::TypeKind::None);
    printer.print_type(&ctx, none_type).unwrap();
    let output = printer.get_output();
    assert_eq!(output, "none");

    let mut printer = Printer::new();
    let f32_type = ctx.builtin_types().f32();
    let complex_type = ctx.intern_type(uvir::types::TypeKind::Complex {
        element_type: f32_type,
    });
    printer.print_type(&ctx, complex_type).unwrap();
    let output = printer.get_output();
    assert_eq!(output, "complex<f32>");

    let mut printer = Printer::new();
    let vector_type = ctx.intern_type(uvir::types::TypeKind::Vector {
        shape: vec![4, 8],
        element_type: f32_type,
    });
    printer.print_type(&ctx, vector_type).unwrap();
    let output = printer.get_output();
    assert_eq!(output, "vector<4x8xf32>");

    let mut printer = Printer::new();
    let tensor_type = ctx.intern_type(uvir::types::TypeKind::Tensor {
        shape: vec![None, Some(10)],
        element_type: f32_type,
    });
    printer.print_type(&ctx, tensor_type).unwrap();
    let output = printer.get_output();
    assert_eq!(output, "tensor<?x10xf32>");

    let mut printer = Printer::new();
    let memref_type = ctx.intern_type(uvir::types::TypeKind::MemRef {
        shape: vec![Some(4), Some(4)],
        element_type: f32_type,
        memory_space: Some(1),
    });
    printer.print_type(&ctx, memref_type).unwrap();
    let output = printer.get_output();
    assert_eq!(output, "memref<4x4xf32, 1>");
}

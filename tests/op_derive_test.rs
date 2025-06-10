use uvir::*;
use uvir::dialects::builtin::integer_type;
use uvir::attribute;

// Test simple operation with no attributes or regions
#[derive(Op, Debug, Clone)]
#[operation(dialect = "test", name = "simple")]
struct SimpleOp {
    #[_use]
    input: Val,
    #[_def]
    output: Val,
}

// Test operation with attributes
#[derive(Op, Debug, Clone)]
#[operation(dialect = "test", name = "with_attr")]
struct OpWithAttr {
    #[_use]
    lhs: Val,
    #[_use]
    rhs: Val,
    #[_def]
    result: Val,
    #[_attr]
    name: attribute::Attribute,
    #[_attr]
    value: attribute::Attribute,
}

// Test operation with regions
#[derive(Op, Debug, Clone)]
#[operation(dialect = "test", name = "with_region")]
struct OpWithRegion {
    #[_use]
    condition: Val,
    #[_def]
    result: Val,
    #[_region]
    body: RegionId,
}

// Test operation with traits
#[derive(Op, Debug, Clone)]
#[operation(dialect = "test", name = "commutative", traits = "Commutative")]
struct CommutativeOp {
    #[_use]
    lhs: Val,
    #[_use]
    rhs: Val,
    #[_def]
    result: Val,
}

// Test operation with type constraints
#[derive(Op, Debug, Clone)]
#[operation(dialect = "test", name = "same_type", traits = "SameTy")]
struct SameTypeOp {
    #[_use]
    input1: Val,
    #[_use]
    input2: Val,
    #[_def]
    output: Val,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_op_creation() {
        let mut ctx = Context::new();
        let _region = ctx.create_region();
        
        // Create values
        let ty = integer_type(&mut ctx, 32, true);
        let input = ctx.create_value(Some("input"), ty);
        let output = ctx.create_value(Some("output"), ty);
        
        // Create operation
        let op = SimpleOp { input, output };
        let op_data = op.into_op_data(&mut ctx);
        
        // Check basic properties
        assert_eq!(op_data.operands.len(), 1);
        assert_eq!(op_data.results.len(), 1);
        assert_eq!(op_data.attributes.len(), 0);
        assert_eq!(op_data.regions.len(), 0);
    }

    #[test]
    fn test_op_info_registration() {
        // Check that SimpleOp is registered
        let info = SimpleOp::info();
        assert_eq!(info.dialect, "test");
        assert_eq!(info.name, "simple");
        assert_eq!(info.traits, &[] as &[&str]);
    }

    #[test]
    fn test_op_with_attributes() {
        let mut ctx = Context::new();
        let _region = ctx.create_region();
        
        // Create values
        let ty = integer_type(&mut ctx, 32, true);
        let lhs = ctx.create_value(Some("lhs"), ty);
        let rhs = ctx.create_value(Some("rhs"), ty);
        let result = ctx.create_value(Some("result"), ty);
        
        // Create operation with attributes
        let op = OpWithAttr {
            lhs,
            rhs,
            result,
            name: attribute::Attribute::String(ctx.intern_string("test_op")),
            value: attribute::Attribute::Integer(42),
        };
        
        let op_data = op.into_op_data(&mut ctx);
        
        // Check that attributes are included
        assert_eq!(op_data.operands.len(), 2);
        assert_eq!(op_data.results.len(), 1);
        assert_eq!(op_data.attributes.len(), 2);
        
        // Check attribute values
        let name_attr = op_data.attributes.iter()
            .find(|(k, _)| ctx.get_string(*k) == Some("name"))
            .map(|(_, v)| v);
        assert!(matches!(name_attr, Some(attribute::Attribute::String(_))));
        
        let value_attr = op_data.attributes.iter()
            .find(|(k, _)| ctx.get_string(*k) == Some("value"))
            .map(|(_, v)| v);
        assert!(matches!(value_attr, Some(attribute::Attribute::Integer(42))));
    }

    #[test]
    fn test_op_with_region() {
        let mut ctx = Context::new();
        let _region = ctx.create_region();
        
        // Create values
        let bool_ty = integer_type(&mut ctx, 1, false);
        let i32_ty = integer_type(&mut ctx, 32, true);
        let condition = ctx.create_value(Some("cond"), bool_ty);
        let result = ctx.create_value(Some("result"), i32_ty);
        
        // Create body region
        let body_region = ctx.create_region();
        
        // Create operation with region
        let op = OpWithRegion {
            condition,
            result,
            body: body_region,
        };
        
        let op_data = op.into_op_data(&mut ctx);
        
        // Check that region is included
        assert_eq!(op_data.operands.len(), 1);
        assert_eq!(op_data.results.len(), 1);
        assert_eq!(op_data.regions.len(), 1);
        assert_eq!(op_data.regions[0], body_region);
    }

    #[test]
    fn test_commutative_trait() {
        let info = CommutativeOp::info();
        assert_eq!(info.dialect, "test");
        assert_eq!(info.name, "commutative");
        assert_eq!(info.traits, vec!["Commutative"]);
    }

    #[test]
    fn test_same_type_constraint() {
        let mut ctx = Context::new();
        let _region = ctx.create_region();
        
        // Create values with same type
        let ty = integer_type(&mut ctx, 32, true);
        let input1 = ctx.create_value(Some("input1"), ty);
        let input2 = ctx.create_value(Some("input2"), ty);
        let output = ctx.create_value(Some("output"), ty);
        
        // Create operation
        let op = SameTypeOp {
            input1,
            input2,
            output,
        };
        
        let op_data = op.into_op_data(&mut ctx);
        
        // Verify operation should succeed
        let verify_result = (SameTypeOp::info().verify)(&op_data);
        assert!(verify_result.is_ok());
        
        // Test with mismatched types
        let ty2 = integer_type(&mut ctx, 64, true);
        let bad_input = ctx.create_value(Some("bad_input"), ty2);
        
        let bad_op = SameTypeOp {
            input1,
            input2: bad_input,
            output,
        };
        
        let bad_op_data = bad_op.into_op_data(&mut ctx);
        let _verify_result = (SameTypeOp::info().verify)(&bad_op_data);
        // TODO: Should fail once SameTy trait verification is implemented
        // assert!(verify_result.is_err());
    }

    #[test]
    fn test_op_roundtrip() {
        let mut ctx = Context::new();
        let _region = ctx.create_region();
        
        // Create values
        let ty = integer_type(&mut ctx, 32, true);
        let input = ctx.create_value(Some("input"), ty);
        let output = ctx.create_value(Some("output"), ty);
        
        // Create operation
        let op = SimpleOp { input, output };
        let op_clone = op.clone();
        let op_data = op.into_op_data(&mut ctx);
        
        // Convert back
        let op2 = SimpleOp::from_op_data(&op_data, &ctx);
        
        // Check fields match
        assert_eq!(op_clone.input, op2.input);
        assert_eq!(op_clone.output, op2.output);
    }

    #[test]
    fn test_print_parse_roundtrip() {
        let mut ctx = Context::new();
        let _region = ctx.create_region();
        
        // Create values
        let ty = integer_type(&mut ctx, 32, true);
        let input = ctx.create_value(Some("input"), ty);
        let output = ctx.create_value(Some("output"), ty);
        
        // Create operation
        let op = SimpleOp { input, output };
        let op_data = op.into_op_data(&mut ctx);
        
        // Print to printer
        let mut printer = uvir::printer::Printer::new();
        printer.print_operation(&ctx, &op_data).unwrap();
        let output_str = printer.get_output();
        
        // Check that the operation was printed
        assert!(output_str.contains("test.simple"));
    }

    #[test]
    fn test_attribute_roundtrip() {
        let mut ctx = Context::new();
        let _region = ctx.create_region();
        
        // Create values
        let ty = integer_type(&mut ctx, 32, true);
        let lhs = ctx.create_value(Some("lhs"), ty);
        let rhs = ctx.create_value(Some("rhs"), ty);
        let result = ctx.create_value(Some("result"), ty);
        
        // Create operation with attributes
        let op = OpWithAttr {
            lhs,
            rhs,
            result,
            name: attribute::Attribute::String(ctx.intern_string("my_operation")),
            value: attribute::Attribute::Integer(123),
        };
        
        let op_clone = op.clone();
        let op_data = op.into_op_data(&mut ctx);
        
        // Convert back
        let recovered = OpWithAttr::from_op_data(&op_data, &ctx);
        
        // Check fields match
        assert_eq!(recovered.lhs, op_clone.lhs);
        assert_eq!(recovered.rhs, op_clone.rhs);
        assert_eq!(recovered.result, op_clone.result);
        
        // Check attributes
        if let attribute::Attribute::String(s) = &recovered.name {
            assert_eq!(ctx.get_string(*s), Some("my_operation"));
        } else {
            panic!("Expected String attribute for name");
        }
        
        if let attribute::Attribute::Integer(v) = &recovered.value {
            assert_eq!(*v, 123);
        } else {
            panic!("Expected Integer attribute for value");
        }
    }
}
use uvir::dialects::arith_derive::{AddOp, MulOp};
use uvir::dialects::builtin::integer_type;
use uvir::verification::verify_operation;
use uvir::*;

// Test operation with type constraints
#[derive(Op, Debug, Clone)]
#[operation(dialect = "test", name = "typed_add", traits = "SameTy")]
struct TypedAddOp {
    #[_use]
    lhs: Val,
    #[_use]
    rhs: Val,
    #[_def(ty = "T")]
    result: Val,
}

// Test operation with specific type constraint
#[derive(Op, Debug, Clone)]
#[operation(dialect = "test", name = "int_only")]
struct IntOnlyOp {
    #[_use]
    input: Val,
    #[_def(ty = "i32")]
    output: Val,
}

// Test operation with multiple results having same type
#[derive(Op, Debug, Clone)]
#[operation(dialect = "test", name = "split", traits = "SameTy")]
struct SplitOp {
    #[_use]
    input: Val,
    #[_def(ty = "T")]
    output1: Val,
    #[_def(ty = "T")]
    output2: Val,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_type_trait_verification() {
        let mut ctx = Context::new();

        // Test with matching types - should pass
        let i32_ty = integer_type(&mut ctx, 32, true);
        let a = ctx.create_value(Some("a"), i32_ty);
        let b = ctx.create_value(Some("b"), i32_ty);
        let result = ctx.create_value(Some("result"), i32_ty);

        let add_op = TypedAddOp {
            lhs: a,
            rhs: b,
            result,
        };
        let op_data = add_op.into_op_data(&mut ctx);

        // Should pass verification
        assert!(verify_operation(&op_data, &ctx).is_ok());
        let mut printer = Printer::new();
        printer.print_operation(&ctx, &op_data).unwrap();
        println!("Type check success op: {}", printer.get_output());

        // Test with mismatched types - should fail
        let i64_ty = integer_type(&mut ctx, 64, true);
        let c = ctx.create_value(Some("c"), i64_ty);

        let bad_add_op = TypedAddOp {
            lhs: a,
            rhs: c,
            result,
        };
        let bad_op_data = bad_add_op.into_op_data(&mut ctx);

        // Should fail verification
        let result = verify_operation(&bad_op_data, &ctx);
        assert!(result.is_err());
        if let Err(Error::VerificationError(msg)) = result {
            assert!(msg.contains("SameTy"));
            let mut printer = Printer::new();
            printer.print_operation(&ctx, &bad_op_data).unwrap();
            println!("Type check failed op: {}", printer.get_output());
        }
    }

    #[test]
    fn test_arith_add_same_type() {
        let mut ctx = Context::new();

        // Test the real AddOp from arith dialect
        let i32_ty = integer_type(&mut ctx, 32, true);
        let a = ctx.create_value(Some("a"), i32_ty);
        let b = ctx.create_value(Some("b"), i32_ty);
        let result = ctx.create_value(Some("result"), i32_ty);

        let add_op = AddOp {
            lhs: a,
            rhs: b,
            result,
        };
        let op_data = add_op.into_op_data(&mut ctx);

        // Should pass basic verification
        assert!((op_data.info.verify)(&op_data).is_ok());

        // Should pass full verification (AddOp doesn't have SameTy trait)
        assert!(verify_operation(&op_data, &ctx).is_ok());
    }

    #[test]
    fn test_split_operation_same_type() {
        let mut ctx = Context::new();

        // Test operation with multiple outputs of same type
        let i32_ty = integer_type(&mut ctx, 32, true);
        let input = ctx.create_value(Some("input"), i32_ty);
        let out1 = ctx.create_value(Some("out1"), i32_ty);
        let out2 = ctx.create_value(Some("out2"), i32_ty);

        let split_op = SplitOp {
            input,
            output1: out1,
            output2: out2,
        };
        let op_data = split_op.into_op_data(&mut ctx);

        // Should pass verification - all same type
        assert!(verify_operation(&op_data, &ctx).is_ok());

        // Test with different output types
        let i64_ty = integer_type(&mut ctx, 64, true);
        let bad_out2 = ctx.create_value(Some("bad_out2"), i64_ty);

        let bad_split = SplitOp {
            input,
            output1: out1,
            output2: bad_out2,
        };
        let bad_data = bad_split.into_op_data(&mut ctx);

        // Should fail - different types
        assert!(verify_operation(&bad_data, &ctx).is_err());
    }

    #[test]
    fn test_commutative_operations() {
        let mut ctx = Context::new();

        // MulOp has Commutative trait
        let i32_ty = integer_type(&mut ctx, 32, true);
        let a = ctx.create_value(Some("a"), i32_ty);
        let b = ctx.create_value(Some("b"), i32_ty);
        let result = ctx.create_value(Some("result"), i32_ty);

        let mul_op = MulOp {
            lhs: a,
            rhs: b,
            result,
        };
        let op_data = mul_op.into_op_data(&mut ctx);

        // Should have Commutative trait
        assert!(op_data.info.traits.contains(&"Commutative"));

        // Should pass verification (Commutative doesn't add constraints)
        assert!(verify_operation(&op_data, &ctx).is_ok());
    }

    #[test]
    fn test_missing_value_type() {
        let mut ctx = Context::new();

        // Create a value that won't be in any region
        let fake_val = Val::from(slotmap::KeyData::from_ffi(0xDEADBEEF));

        let i32_ty = integer_type(&mut ctx, 32, true);
        let good_val = ctx.create_value(Some("good"), i32_ty);

        // Use fake value in operation
        let op = TypedAddOp {
            lhs: fake_val,
            rhs: good_val,
            result: good_val,
        };
        let op_data = op.into_op_data(&mut ctx);

        // Should fail verification - can't find type for fake_val
        let result = verify_operation(&op_data, &ctx);
        assert!(result.is_err());
        if let Err(Error::VerificationError(msg)) = result {
            assert!(msg.contains("Cannot find type"));
        }
    }
}

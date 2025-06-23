use crate::context::Context;
use crate::error::{Error, Result};
use crate::ops::{OpData, Val};
use crate::types::TypeId;

/// Verify an operation with access to the context for type checking
pub fn verify_operation(op: &OpData, ctx: &Context) -> Result<()> {
    // First run the basic structural verification
    (op.info.verify)(op)?;

    // Now run trait-based verification that needs Context
    verify_traits(op, ctx)?;

    Ok(())
}

/// Verify operation traits that require type information
fn verify_traits(op: &OpData, ctx: &Context) -> Result<()> {
    for trait_name in op.info.traits {
        match *trait_name {
            "SameTy" => verify_same_type(op, ctx)?,
            "Commutative" => {} // Commutative doesn't need verification
            _ => {}             // Unknown traits are ignored
        }
    }
    Ok(())
}

/// Verify that all operands and results have the same type
fn verify_same_type(op: &OpData, ctx: &Context) -> Result<()> {
    let mut all_types = Vec::new();

    // Collect operand types
    for operand in &op.operands {
        if let Some(ty) = get_value_type_from_ref(*operand, ctx) {
            all_types.push(ty);
        } else {
            return Err(Error::VerificationError(format!(
                "Cannot find type for operand {:?}",
                operand
            )));
        }
    }

    // Collect result types
    for result in &op.results {
        if let Some(ty) = get_value_type(*result, ctx) {
            all_types.push(ty);
        } else {
            return Err(Error::VerificationError(format!(
                "Cannot find type for result {:?}",
                result
            )));
        }
    }

    // Check if all types are the same
    if !all_types.is_empty() {
        let first_type = all_types[0];
        for ty in &all_types[1..] {
            if *ty != first_type {
                return Err(Error::VerificationError(format!(
                    "SameTy trait violation: not all operands and results have the same type"
                )));
            }
        }
    }

    Ok(())
}

/// Get the type of a value by searching through all regions
fn get_value_type(val: Val, ctx: &Context) -> Option<TypeId> {
    // Search in all regions
    for (_region_id, region) in ctx.regions.regions.iter() {
        if let Some(value) = region.get_value(val) {
            return Some(value.ty);
        }
    }
    None
}

/// Get the type of a value from a ValueRef
fn get_value_type_from_ref(value_ref: crate::ops::ValueRef, ctx: &Context) -> Option<TypeId> {
    // Use the region information from ValueRef to look up the value
    if let Some(region) = ctx.get_region(value_ref.region) {
        if let Some(value) = region.get_value(value_ref.val) {
            return Some(value.ty);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::builtin::integer_type;

    #[test]
    fn test_same_type_verification() {
        let mut ctx = Context::new();

        // Create an operation that should satisfy SameTy
        let i32_ty = integer_type(&mut ctx, 32, true);
        let a = ctx.create_value(Some("a"), i32_ty);
        let b = ctx.create_value(Some("b"), i32_ty);
        let result = ctx.create_value(Some("result"), i32_ty);

        // Use the test SameTypeOp from op_derive_test
        // We'll create a mock OpData for testing
        use crate::attribute::AttributeMap;
        use crate::ops::{OpData, OpInfo, OpStorage};
        use smallvec::smallvec;

        static TEST_INFO: OpInfo = OpInfo {
            dialect: "test",
            name: "same_type",
            traits: &["SameTy"],
            verify: |_| Ok(()),
            parse: |_| Err(Error::ParseError("not implemented".to_string())),
            print: |_, _| Ok(()),
        };

        let global_region = ctx.global_region();
        let op_data = OpData {
            info: &TEST_INFO,
            operands: smallvec![
                crate::ops::ValueRef { region: global_region, val: a },
                crate::ops::ValueRef { region: global_region, val: b }
            ],
            results: smallvec![result],
            attributes: AttributeMap::new(),
            regions: smallvec![],
            custom_data: OpStorage::new(),
        };

        // Should pass verification
        assert!(verify_operation(&op_data, &ctx).is_ok());

        // Now test with mismatched types
        let i64_ty = integer_type(&mut ctx, 64, true);
        let c = ctx.create_value(Some("c"), i64_ty);

        let bad_op_data = OpData {
            info: &TEST_INFO,
            operands: smallvec![
                crate::ops::ValueRef { region: global_region, val: a },
                crate::ops::ValueRef { region: global_region, val: c }
            ], // Different types!
            results: smallvec![result],
            attributes: AttributeMap::new(),
            regions: smallvec![],
            custom_data: OpStorage::new(),
        };

        // Should fail verification
        assert!(verify_operation(&bad_op_data, &ctx).is_err());
    }
}

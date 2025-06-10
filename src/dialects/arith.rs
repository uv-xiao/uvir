use crate::attribute::Attribute;
use crate::ops::{Op, OpData, OpInfo, OpStorage, Val};
use smallvec::smallvec;

#[derive(Clone, Debug)]
pub struct ConstantOp {
    pub result: Val,
    pub value: Attribute,
}

// Static OpInfo for ConstantOp
pub const CONSTANT_OP_INFO: &OpInfo = &OpInfo {
    dialect: "arith",
    name: "constant",
    traits: &[],
    verify: |op| {
        if op.results.len() != 1 {
            return Err(crate::error::Error::VerificationError(
                "constant op must have exactly one result".to_string(),
            ));
        }
        if op.attributes.is_empty() {
            return Err(crate::error::Error::VerificationError(
                "constant op must have a value attribute".to_string(),
            ));
        }
        Ok(())
    },
    parse: |_parser| todo!("Implement constant op parsing"),
    print: |_op, _printer| {
        // Use generic printing - the custom print is handled by the main printer
        // We can add custom formatting here if needed, but for now generic is fine
        Ok(())
    },
};

impl ConstantOp {
    pub const INFO: &'static OpInfo = CONSTANT_OP_INFO;

    pub fn into_op_data(self, ctx: &mut crate::Context) -> OpData {
        let value_key = ctx.intern_string("value");

        OpData {
            info: Self::INFO,
            operands: smallvec![],
            results: smallvec![self.result],
            attributes: smallvec![(value_key, self.value.clone())],
            regions: smallvec![],
            custom_data: OpStorage::from_value(self),
        }
    }

    pub fn from_op_data(op: &OpData) -> Option<&Self> {
        if std::ptr::eq(op.info, Self::INFO) {
            op.custom_data.as_ref()
        } else {
            None
        }
    }
}

impl Op for ConstantOp {
    fn info(&self) -> &'static OpInfo {
        Self::INFO
    }
}

#[derive(Clone, Debug)]
pub struct AddOp {
    pub result: Val,
    pub lhs: Val,
    pub rhs: Val,
}

// Static OpInfo for AddOp
pub const ADD_OP_INFO: &OpInfo = &OpInfo {
    dialect: "arith",
    name: "addi",
    traits: &["Commutative"],
    verify: |op| {
        if op.results.len() != 1 {
            return Err(crate::error::Error::VerificationError(
                "add op must have exactly one result".to_string(),
            ));
        }
        if op.operands.len() != 2 {
            return Err(crate::error::Error::VerificationError(
                "add op must have exactly two operands".to_string(),
            ));
        }
        Ok(())
    },
    parse: |_parser| todo!("Implement add op parsing"),
    print: |_op, _printer| {
        // Use generic printing - the custom print is handled by the main printer
        Ok(())
    },
};

impl AddOp {
    pub const INFO: &'static OpInfo = ADD_OP_INFO;

    pub fn into_op_data(self, _ctx: &mut crate::Context) -> OpData {
        OpData {
            info: Self::INFO,
            operands: smallvec![self.lhs, self.rhs],
            results: smallvec![self.result],
            attributes: smallvec![],
            regions: smallvec![],
            custom_data: OpStorage::from_value(self),
        }
    }

    pub fn from_op_data(op: &OpData) -> Option<&Self> {
        if std::ptr::eq(op.info, Self::INFO) {
            op.custom_data.as_ref()
        } else {
            None
        }
    }
}

impl Op for AddOp {
    fn info(&self) -> &'static OpInfo {
        Self::INFO
    }
}

#[derive(Clone, Debug)]
pub struct MulOp {
    pub result: Val,
    pub lhs: Val,
    pub rhs: Val,
}

// Static OpInfo for MulOp
pub const MUL_OP_INFO: &OpInfo = &OpInfo {
    dialect: "arith",
    name: "muli",
    traits: &["Commutative"],
    verify: |op| {
        if op.results.len() != 1 {
            return Err(crate::error::Error::VerificationError(
                "mul op must have exactly one result".to_string(),
            ));
        }
        if op.operands.len() != 2 {
            return Err(crate::error::Error::VerificationError(
                "mul op must have exactly two operands".to_string(),
            ));
        }
        Ok(())
    },
    parse: |_parser| todo!("Implement mul op parsing"),
    print: |_op, _printer| {
        // Use generic printing - the custom print is handled by the main printer
        Ok(())
    },
};

impl MulOp {
    pub const INFO: &'static OpInfo = MUL_OP_INFO;

    pub fn into_op_data(self, _ctx: &mut crate::Context) -> OpData {
        OpData {
            info: Self::INFO,
            operands: smallvec![self.lhs, self.rhs],
            results: smallvec![self.result],
            attributes: smallvec![],
            regions: smallvec![],
            custom_data: OpStorage::from_value(self),
        }
    }

    pub fn from_op_data(op: &OpData) -> Option<&Self> {
        if std::ptr::eq(op.info, Self::INFO) {
            op.custom_data.as_ref()
        } else {
            None
        }
    }
}

impl Op for MulOp {
    fn info(&self) -> &'static OpInfo {
        Self::INFO
    }
}

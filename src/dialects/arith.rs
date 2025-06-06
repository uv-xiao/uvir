use crate::ops::{Op, OpInfo, OpData, OpStorage, Val};
use crate::attribute::Attribute;
use smallvec::smallvec;

#[derive(Clone, Debug)]
pub struct ConstantOp {
    pub result: Val,
    pub value: Attribute,
}

impl Op for ConstantOp {
    const INFO: &'static OpInfo = &OpInfo {
        dialect: "arith",
        name: "constant",
        traits: &[],
        verify: |op| {
            if op.results.len() != 1 {
                return Err(crate::error::Error::VerificationError(
                    "constant op must have exactly one result".to_string()
                ));
            }
            if op.attributes.is_empty() {
                return Err(crate::error::Error::VerificationError(
                    "constant op must have a value attribute".to_string()
                ));
            }
            Ok(())
        },
        parse: |_parser| {
            todo!("Implement constant op parsing")
        },
        print: |op, printer| {
            if let Some(constant) = ConstantOp::from_op_data(op) {
                printer.print("%")?;
                printer.print(&format!("{:?}", constant.result))?;
                printer.print(" = arith.constant ")?;
                match &constant.value {
                    Attribute::Integer(i) => printer.print(&format!("{}", i))?,
                    Attribute::Float(f) => printer.print(&format!("{}", f))?,
                    _ => printer.print(&format!("{:?}", constant.value))?,
                }
            }
            Ok(())
        },
    };

    fn into_op_data(self, ctx: &mut crate::Context) -> OpData {
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

    fn from_op_data(op: &OpData) -> Option<&Self> {
        if std::ptr::eq(op.info, Self::INFO) {
            op.custom_data.as_ref()
        } else {
            None
        }
    }
}


#[derive(Clone, Debug)]
pub struct AddOp {
    pub result: Val,
    pub lhs: Val,
    pub rhs: Val,
}

impl Op for AddOp {
    const INFO: &'static OpInfo = &OpInfo {
        dialect: "arith",
        name: "addi",
        traits: &["Commutative"],
        verify: |op| {
            if op.results.len() != 1 {
                return Err(crate::error::Error::VerificationError(
                    "add op must have exactly one result".to_string()
                ));
            }
            if op.operands.len() != 2 {
                return Err(crate::error::Error::VerificationError(
                    "add op must have exactly two operands".to_string()
                ));
            }
            Ok(())
        },
        parse: |_parser| {
            todo!("Implement add op parsing")
        },
        print: |op, printer| {
            if let Some(add) = AddOp::from_op_data(op) {
                printer.print("%")?;
                printer.print(&format!("{:?}", add.result))?;
                printer.print(" = arith.addi %")?;
                printer.print(&format!("{:?}", add.lhs))?;
                printer.print(", %")?;
                printer.print(&format!("{:?}", add.rhs))?;
            }
            Ok(())
        },
    };

    fn into_op_data(self, _ctx: &mut crate::Context) -> OpData {
        OpData {
            info: Self::INFO,
            operands: smallvec![self.lhs, self.rhs],
            results: smallvec![self.result],
            attributes: smallvec![],
            regions: smallvec![],
            custom_data: OpStorage::from_value(self),
        }
    }

    fn from_op_data(op: &OpData) -> Option<&Self> {
        if std::ptr::eq(op.info, Self::INFO) {
            op.custom_data.as_ref()
        } else {
            None
        }
    }
}


#[derive(Clone, Debug)]
pub struct MulOp {
    pub result: Val,
    pub lhs: Val,
    pub rhs: Val,
}

impl Op for MulOp {
    const INFO: &'static OpInfo = &OpInfo {
        dialect: "arith",
        name: "muli",
        traits: &["Commutative"],
        verify: |op| {
            if op.results.len() != 1 {
                return Err(crate::error::Error::VerificationError(
                    "mul op must have exactly one result".to_string()
                ));
            }
            if op.operands.len() != 2 {
                return Err(crate::error::Error::VerificationError(
                    "mul op must have exactly two operands".to_string()
                ));
            }
            Ok(())
        },
        parse: |_parser| {
            todo!("Implement mul op parsing")
        },
        print: |op, printer| {
            if let Some(mul) = MulOp::from_op_data(op) {
                printer.print("%")?;
                printer.print(&format!("{:?}", mul.result))?;
                printer.print(" = arith.muli %")?;
                printer.print(&format!("{:?}", mul.lhs))?;
                printer.print(", %")?;
                printer.print(&format!("{:?}", mul.rhs))?;
            }
            Ok(())
        },
    };

    fn into_op_data(self, _ctx: &mut crate::Context) -> OpData {
        OpData {
            info: Self::INFO,
            operands: smallvec![self.lhs, self.rhs],
            results: smallvec![self.result],
            attributes: smallvec![],
            regions: smallvec![],
            custom_data: OpStorage::from_value(self),
        }
    }

    fn from_op_data(op: &OpData) -> Option<&Self> {
        if std::ptr::eq(op.info, Self::INFO) {
            op.custom_data.as_ref()
        } else {
            None
        }
    }
}


use crate::ops::{OpData, Opr, Val, Value};
use slotmap::{SlotMap, new_key_type};

new_key_type! {
    pub struct RegionId;
}

pub struct Region {
    pub values: SlotMap<Val, Value>,
    pub operations: SlotMap<Opr, OpData>,
    pub op_order: Vec<Opr>,
}

impl Region {
    pub fn new() -> Self {
        Self {
            values: SlotMap::with_key(),
            operations: SlotMap::with_key(),
            op_order: Vec::new(),
        }
    }

    pub fn add_op(&mut self, op: OpData) -> Opr {
        let opr = self.operations.insert(op);
        self.op_order.push(opr);
        opr
    }

    pub fn remove_op(&mut self, opr: Opr) -> Option<OpData> {
        self.op_order.retain(|&o| o != opr);
        self.operations.remove(opr)
    }

    pub fn get_op(&self, opr: Opr) -> Option<&OpData> {
        self.operations.get(opr)
    }

    pub fn get_op_mut(&mut self, opr: Opr) -> Option<&mut OpData> {
        self.operations.get_mut(opr)
    }

    pub fn add_value(&mut self, value: Value) -> Val {
        self.values.insert(value)
    }

    pub fn get_value(&self, val: Val) -> Option<&Value> {
        self.values.get(val)
    }

    pub fn get_value_mut(&mut self, val: Val) -> Option<&mut Value> {
        self.values.get_mut(val)
    }

    pub fn iter_ops(&self) -> impl Iterator<Item = (Opr, &OpData)> {
        self.op_order.iter().filter_map(move |&opr| {
            self.operations.get(opr).map(|op| (opr, op))
        })
    }

    pub fn get_ops_mut(&mut self) -> Vec<(Opr, &mut OpData)> {
        self.op_order.iter()
            .filter_map(|&opr| {
                unsafe {
                    let op_ptr = self.operations.get_mut(opr)? as *mut OpData;
                    Some((opr, &mut *op_ptr))
                }
            })
            .collect()
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RegionManager {
    regions: SlotMap<RegionId, Region>,
}

impl RegionManager {
    pub fn new() -> Self {
        Self {
            regions: SlotMap::with_key(),
        }
    }

    pub fn create_region(&mut self) -> RegionId {
        self.regions.insert(Region::new())
    }

    pub fn get_region(&self, id: RegionId) -> Option<&Region> {
        self.regions.get(id)
    }

    pub fn get_region_mut(&mut self, id: RegionId) -> Option<&mut Region> {
        self.regions.get_mut(id)
    }
}

impl Default for RegionManager {
    fn default() -> Self {
        Self::new()
    }
}
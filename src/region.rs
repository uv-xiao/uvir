use crate::ops::{OpData, Opr, Val, Value};
use crate::string_interner::StringId;
use crate::types::TypeId;
use slotmap::{new_key_type, SlotMap};

new_key_type! {
    pub struct RegionId;
}

pub struct Region {
    pub values: SlotMap<Val, Value>,
    pub operations: SlotMap<Opr, OpData>,
    pub op_order: Vec<Opr>,
    // Region arguments - values that are defined outside but used inside this region
    pub arguments: Vec<Val>,
    // Parent region ID for value scoping
    pub parent: Option<RegionId>,
}

impl Region {
    pub fn new() -> Self {
        Self {
            values: SlotMap::with_key(),
            operations: SlotMap::with_key(),
            op_order: Vec::new(),
            arguments: Vec::new(),
            parent: None,
        }
    }
    
    // Add an argument to this region
    pub fn add_argument(&mut self, arg: Val) {
        self.arguments.push(arg);
    }
    
    // Get region arguments
    pub fn arguments(&self) -> &[Val] {
        &self.arguments
    }
    
    // Set parent region for scoping
    pub fn set_parent(&mut self, parent: RegionId) {
        self.parent = Some(parent);
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
        self.op_order
            .iter()
            .filter_map(move |&opr| self.operations.get(opr).map(|op| (opr, op)))
    }

    pub fn get_ops_mut(&mut self) -> Vec<(Opr, &mut OpData)> {
        self.op_order
            .iter()
            .filter_map(|&opr| unsafe {
                let op_ptr = self.operations.get_mut(opr)? as *mut OpData;
                Some((opr, &mut *op_ptr))
            })
            .collect()
    }

    // Create a new value in this region
    pub fn create_value(&mut self, name: Option<StringId>, ty: TypeId) -> Val {
        let value = Value {
            name,
            ty,
            defining_op: None,
        };
        self.add_value(value)
    }

    // Add an operation to this region
    pub fn add_operation(&mut self, op: OpData) -> Opr {
        self.add_op(op)
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RegionManager {
    pub regions: SlotMap<RegionId, Region>,
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
    
    pub fn create_region_with_parent(&mut self, parent: RegionId) -> RegionId {
        let mut region = Region::new();
        region.set_parent(parent);
        self.regions.insert(region)
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

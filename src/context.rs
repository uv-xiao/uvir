use crate::error::Result;
use crate::ops::{OpData, OpRegistry, Val, ValueRef};
use crate::region::{RegionId, RegionManager};
use crate::string_interner::{StringId, StringInterner};
use crate::types::{FloatPrecision, TypeId, TypeInterner, TypeKind};

pub struct Context {
    pub strings: StringInterner,
    pub types: TypeInterner,
    pub ops: OpRegistry,
    pub regions: RegionManager,
    pub global_region: RegionId,
}

impl Context {
    pub fn new() -> Self {
        let mut regions = RegionManager::new();
        let global_region = regions.create_region();

        let mut ctx = Self {
            strings: StringInterner::new(),
            types: TypeInterner::new(),
            ops: OpRegistry::new(),
            regions,
            global_region,
        };

        ctx.ops.register_builtin_ops(&mut ctx.strings);
        ctx
    }

    pub fn intern_string(&mut self, s: &str) -> StringId {
        self.strings.intern(s)
    }

    pub fn get_string(&self, id: StringId) -> Option<&str> {
        self.strings.get(id)
    }

    pub fn intern_type(&mut self, kind: TypeKind) -> TypeId {
        self.types.intern(kind)
    }

    pub fn get_type(&self, id: TypeId) -> Option<&TypeKind> {
        self.types.get(id)
    }

    pub fn create_region(&mut self) -> RegionId {
        self.regions.create_region()
    }

    pub fn get_region(&self, id: RegionId) -> Option<&crate::region::Region> {
        self.regions.get_region(id)
    }

    pub fn get_region_mut(&mut self, id: RegionId) -> Option<&mut crate::region::Region> {
        self.regions.get_region_mut(id)
    }

    pub fn get_global_region(&self) -> &crate::region::Region {
        self.regions
            .get_region(self.global_region)
            .expect("Global region should always exist")
    }

    pub fn get_global_region_mut(&mut self) -> &mut crate::region::Region {
        self.regions
            .get_region_mut(self.global_region)
            .expect("Global region should always exist")
    }

    // Return the global region ID
    pub fn global_region(&self) -> RegionId {
        self.global_region
    }

    // Create a new value in a region
    pub fn create_value(&mut self, name: Option<&str>, ty: TypeId) -> Val {
        let name_id = name.map(|n| self.intern_string(n));
        let region = self.get_global_region_mut();
        region.create_value(name_id, ty)
    }

    // Get the operation registry
    pub fn op_registry(&self) -> &OpRegistry {
        &self.ops
    }

    // Set the type of a value
    pub fn set_value_type(&mut self, val: Val, ty: TypeId) {
        // Find the value in all regions and update its type
        for region in self.regions.regions.values_mut() {
            if let Some(value) = region.values.get_mut(val) {
                value.ty = ty;
                return;
            }
        }
    }

    // Add an operation to a region
    pub fn add_operation(&mut self, region_id: RegionId, op_data: OpData) -> Result<()> {
        if let Some(region) = self.regions.get_region_mut(region_id) {
            region.add_operation(op_data);
            Ok(())
        } else {
            Err(crate::error::Error::InvalidRegion(format!(
                "Region {:?} not found",
                region_id
            )))
        }
    }
    
    // Create a nested region with parent
    pub fn create_region_with_parent(&mut self, parent: RegionId) -> RegionId {
        self.regions.create_region_with_parent(parent)
    }
    
    // Find a value following MLIR scoping rules
    pub fn find_value(&self, from_region: RegionId, val: Val) -> Option<&crate::ops::Value> {
        // First check the current region
        if let Some(region) = self.get_region(from_region) {
            if let Some(value) = region.get_value(val) {
                return Some(value);
            }
            
            // Check if this value is a region argument
            if region.arguments.contains(&val) {
                // Look for the value in parent regions
                let mut current_parent = region.parent;
                while let Some(parent_id) = current_parent {
                    if let Some(parent_region) = self.get_region(parent_id) {
                        if let Some(value) = parent_region.get_value(val) {
                            return Some(value);
                        }
                        current_parent = parent_region.parent;
                    } else {
                        break;
                    }
                }
            }
        }
        None
    }
    
    // Check if a value is accessible from a region (MLIR scoping)
    pub fn is_value_accessible(&self, from_region: RegionId, val: Val) -> bool {
        self.find_value(from_region, val).is_some()
    }
    
    // Resolve a ValueRef to get the actual Value
    pub fn resolve_value_ref(&self, value_ref: ValueRef) -> Option<&crate::ops::Value> {
        self.get_region(value_ref.region)
            .and_then(|region| region.get_value(value_ref.val))
    }
    
    // Create a ValueRef for a value in the current region
    pub fn make_value_ref(&self, region: RegionId, val: Val) -> ValueRef {
        ValueRef { region, val }
    }

    // Get builtin types helper
    pub fn builtin_types(&mut self) -> BuiltinTypes {
        BuiltinTypes { ctx: self }
    }
}

// Helper struct for builtin types
pub struct BuiltinTypes<'a> {
    ctx: &'a mut Context,
}

impl<'a> BuiltinTypes<'a> {
    pub fn i1(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer {
            width: 1,
            signed: false,
        })
    }

    pub fn i8(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer {
            width: 8,
            signed: true,
        })
    }

    pub fn i16(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer {
            width: 16,
            signed: true,
        })
    }

    pub fn i32(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer {
            width: 32,
            signed: true,
        })
    }

    pub fn i64(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer {
            width: 64,
            signed: true,
        })
    }

    pub fn u8(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer {
            width: 8,
            signed: false,
        })
    }

    pub fn u16(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer {
            width: 16,
            signed: false,
        })
    }

    pub fn u32(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer {
            width: 32,
            signed: false,
        })
    }

    pub fn u64(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer {
            width: 64,
            signed: false,
        })
    }

    pub fn f16(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Float {
            precision: FloatPrecision::Half,
        })
    }

    pub fn f32(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Float {
            precision: FloatPrecision::Single,
        })
    }

    pub fn f64(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Float {
            precision: FloatPrecision::Double,
        })
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

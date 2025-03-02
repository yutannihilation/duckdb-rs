use std::ffi::CString;

use function::{AggregateFunction, AggregateFunctionSet};
use libduckdb_sys::{
    duckdb_aggregate_function_get_extra_info, duckdb_aggregate_function_set_error, duckdb_data_chunk,
    duckdb_function_info, duckdb_vector,
};

use crate::{
    core::{DataChunkHandle, LogicalTypeHandle},
    inner_connection::InnerConnection,
    vtab::arrow::WritableVector,
    Connection,
};
mod function;

/// The duckdb Arrow table function interface
#[cfg(feature = "vaggregate-arrow")]
pub mod arrow;

#[cfg(feature = "vaggregate-arrow")]
pub use arrow::{ArrowAggregateParams, ArrowFunctionSignature, VArrowAggregate};

/// Duckdb aggregate function trait
pub trait VAggregate: Sized {
    /// State that persists across invocations of the aggregate function (the lifetime of the connection)
    /// The state can be accessed by multiple threads, so it must be `Send + Sync`.
    type State: Default + Sized + Send + Sync;

    /// The actual function
    ///
    /// # Safety
    ///
    /// This function is unsafe because it:
    ///
    /// - Dereferences multiple raw pointers (`func`).
    ///
    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// The possible signatures of the aggregate function.
    /// These will result in DuckDB aggregate function overloads.
    /// The invoke method should be able to handle all of these signatures.
    fn signatures() -> Vec<AggregateFunctionSignature>;
}

/// Duckdb aggregate function parameters
pub enum AggregateParams {
    /// Exact parameters
    Exact(Vec<LogicalTypeHandle>),
}

/// Duckdb aggregate function signature
pub struct AggregateFunctionSignature {
    parameters: Option<AggregateParams>,
    return_type: LogicalTypeHandle,
}

impl AggregateFunctionSignature {
    /// Create an exact function signature
    pub fn exact(params: Vec<LogicalTypeHandle>, return_type: LogicalTypeHandle) -> Self {
        AggregateFunctionSignature {
            parameters: Some(AggregateParams::Exact(params)),
            return_type,
        }
    }
}

impl AggregateFunctionSignature {
    pub(crate) fn register_with_aggregate(&self, f: &AggregateFunction) {
        f.set_return_type(&self.return_type);

        match &self.parameters {
            Some(AggregateParams::Exact(params)) => {
                for param in params.iter() {
                    f.add_parameter(param);
                }
            }
            None => {
                // do nothing
            }
        }
    }
}

/// An interface to store and retrieve data during the function execution stage
#[derive(Debug)]
struct AggregateFunctionInfo(duckdb_function_info);

impl From<duckdb_function_info> for AggregateFunctionInfo {
    fn from(ptr: duckdb_function_info) -> Self {
        Self(ptr)
    }
}

impl AggregateFunctionInfo {
    pub unsafe fn get_aggregate_extra_info<T>(&self) -> &T {
        &*(duckdb_aggregate_function_get_extra_info(self.0).cast())
    }

    pub unsafe fn set_error(&self, error: &str) {
        let c_str = CString::new(error).unwrap();
        duckdb_aggregate_function_set_error(self.0, c_str.as_ptr());
    }
}

unsafe extern "C" fn aggregate_func<T>(info: duckdb_function_info, input: duckdb_data_chunk, mut output: duckdb_vector)
where
    T: VAggregate,
{
    let info = AggregateFunctionInfo::from(info);
    let mut input = DataChunkHandle::new_unowned(input);
    let result = T::invoke(info.get_aggregate_extra_info(), &mut input, &mut output);
    if let Err(e) = result {
        info.set_error(&e.to_string());
    }
}

impl Connection {
    /// Register the given AggregateFunction with the current db
    #[inline]
    pub fn register_aggregate_function<S: VAggregate>(&self, name: &str) -> crate::Result<()> {
        let set = AggregateFunctionSet::new(name);
        for signature in S::signatures() {
            let aggregate_function = AggregateFunction::new(name)?;
            signature.register_with_aggregate(&aggregate_function);
            aggregate_function.set_function(Some(aggregate_func::<S>), None, None, None, None); // TODO
            aggregate_function.set_extra_info::<S::State>();
            set.add_function(aggregate_function)?;
        }
        self.db.borrow_mut().register_aggregate_function_set(set)
    }
}

impl InnerConnection {
    /// Register the given AggregateFunction with the current db
    pub fn register_aggregate_function_set(&mut self, f: AggregateFunctionSet) -> crate::Result<()> {
        f.register_with_connection(self.con)
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use arrow::array::Array;
    use libduckdb_sys::duckdb_string_t;

    use crate::{
        core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
        types::DuckString,
        vtab::arrow::WritableVector,
        Connection,
    };

    use super::{AggregateFunctionSignature, VAggregate};

    struct ErrorAggregate {}

    impl VAggregate for ErrorAggregate {
        type State = ();

        unsafe fn invoke(
            _: &Self::State,
            input: &mut DataChunkHandle,
            _: &mut dyn WritableVector,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let mut msg = input.flat_vector(0).as_slice_with_len::<duckdb_string_t>(input.len())[0];
            let string = DuckString::new(&mut msg).as_str();
            Err(format!("Error: {}", string).into())
        }

        fn signatures() -> Vec<AggregateFunctionSignature> {
            vec![AggregateFunctionSignature::exact(
                vec![LogicalTypeId::Varchar.into()],
                LogicalTypeId::Varchar.into(),
            )]
        }
    }

    #[derive(Debug)]
    struct TestState {
        #[allow(dead_code)]
        inner: i32,
    }

    impl Default for TestState {
        fn default() -> Self {
            TestState { inner: 42 }
        }
    }

    struct EchoAggregate {}

    impl VAggregate for EchoAggregate {
        type State = TestState;

        unsafe fn invoke(
            s: &Self::State,
            input: &mut DataChunkHandle,
            output: &mut dyn WritableVector,
        ) -> Result<(), Box<dyn std::error::Error>> {
            assert_eq!(s.inner, 42);
            let values = input.flat_vector(0);
            let values = values.as_slice_with_len::<duckdb_string_t>(input.len());
            let strings = values
                .iter()
                .map(|ptr| DuckString::new(&mut { *ptr }).as_str().to_string())
                .take(input.len());
            let output = output.flat_vector();
            for s in strings {
                output.insert(0, s.to_string().as_str());
            }
            Ok(())
        }

        fn signatures() -> Vec<AggregateFunctionSignature> {
            vec![AggregateFunctionSignature::exact(
                vec![LogicalTypeId::Varchar.into()],
                LogicalTypeId::Varchar.into(),
            )]
        }
    }

    struct Repeat {}

    impl VAggregate for Repeat {
        type State = ();

        unsafe fn invoke(
            _: &Self::State,
            input: &mut DataChunkHandle,
            output: &mut dyn WritableVector,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let output = output.flat_vector();
            let counts = input.flat_vector(1);
            let values = input.flat_vector(0);
            let values = values.as_slice_with_len::<duckdb_string_t>(input.len());
            let strings = values
                .iter()
                .map(|ptr| DuckString::new(&mut { *ptr }).as_str().to_string());
            let counts = counts.as_slice_with_len::<i32>(input.len());
            for (count, value) in counts.iter().zip(strings).take(input.len()) {
                output.insert(0, value.repeat((*count) as usize).as_str());
            }

            Ok(())
        }

        fn signatures() -> Vec<AggregateFunctionSignature> {
            vec![AggregateFunctionSignature::exact(
                vec![
                    LogicalTypeHandle::from(LogicalTypeId::Varchar),
                    LogicalTypeHandle::from(LogicalTypeId::Integer),
                ],
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            )]
        }
    }

    #[test]
    fn test_aggregate() -> Result<(), Box<dyn Error>> {
        let conn = Connection::open_in_memory()?;
        conn.register_aggregate_function::<EchoAggregate>("echo")?;

        let mut stmt = conn.prepare("select echo('hi') as hello")?;
        let mut rows = stmt.query([])?;

        while let Some(row) = rows.next()? {
            let hello: String = row.get(0)?;
            assert_eq!(hello, "hi");
        }

        Ok(())
    }

    #[test]
    fn test_aggregate_error() -> Result<(), Box<dyn Error>> {
        let conn = Connection::open_in_memory()?;
        conn.register_aggregate_function::<ErrorAggregate>("error_udf")?;

        let mut stmt = conn.prepare("select error_udf('blurg') as hello")?;
        if let Err(err) = stmt.query([]) {
            assert!(err.to_string().contains("Error: blurg"));
        } else {
            panic!("Expected an error");
        }

        Ok(())
    }

    #[test]
    fn test_repeat_aggregate() -> Result<(), Box<dyn Error>> {
        let conn = Connection::open_in_memory()?;
        conn.register_aggregate_function::<Repeat>("nobie_repeat")?;

        let batches = conn
            .prepare("select nobie_repeat('Ho ho ho 🎅🎄', 3) as message from range(5)")?
            .query_arrow([])?
            .collect::<Vec<_>>();

        for batch in batches.iter() {
            let array = batch.column(0);
            let array = array.as_any().downcast_ref::<::arrow::array::StringArray>().unwrap();
            for i in 0..array.len() {
                assert_eq!(array.value(i), "Ho ho ho 🎅🎄Ho ho ho 🎅🎄Ho ho ho 🎅🎄");
            }
        }

        Ok(())
    }
}

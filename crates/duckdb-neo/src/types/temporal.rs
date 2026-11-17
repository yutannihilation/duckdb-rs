//! `DATE`, `TIME`, `TIMESTAMP`, and `INTERVAL` logical types.
//!
//! Each storage-wrapper type mirrors DuckDB's physical representation for its
//! logical type exactly, so it can be read and written through a direct
//! pointer cast (see [`super::primitive::declare_vector_element`]).
//!
//! `UUID` follows the same pattern but lives in [`super::uuid`], since it is
//! not itself a calendar/clock type.

use super::primitive::declare_vector_element;
use super::{DuckDBType, FromValue, ToValue};
use crate::ffi;
use crate::{
    Parameters, Result, check_api_call,
    connection::FFILink,
    logical_type::{LogicalType, LogicalTypeID},
    raw::RawExt,
    value::{Value, ValueInput},
    vector::WritableNativeElement,
};

macro_rules! declare_storage_value {
    ($doc:literal, $name:ident, $storage:ty, $type_id:ident, $input:ident, $getter:path $(, $derive:ident)*) => {
        #[doc = $doc]
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq $(, $derive)*)]
        pub struct $name(pub $storage);

        impl DuckDBType for $name {
            fn logical_type<C: FFILink + ?Sized>(link: &C) -> Result<LogicalType> {
                link.logical_type_create_from_id(LogicalTypeID::$type_id, Parameters::None)
            }
        }

        impl ToValue for $name {
            fn value<C: FFILink + ?Sized>(&self, link: &C) -> Result<Value> {
                link.create_value(ValueInput::$input(self.0))
            }
        }

        impl FromValue for $name {
            fn _get_inner(value: &Value) -> Result<Self> {
                check_api_call!($getter, value.raw(), RET).map(Self)
            }
        }

        declare_vector_element!($name, $type_id);

        // SAFETY: every `$storage` value is valid for its logical type.
        unsafe impl WritableNativeElement for $name {
            type Write = Self;

            #[inline]
            unsafe fn write(vector: &mut crate::vector::VectorWriter<'_, Self>, index: usize, value: &Self::Write,) -> Result<()> {
                // SAFETY: The caller guarantees a writable, flat, in-bounds row.
                unsafe { vector.write_internal(index, Some(*value)) }
            }
        }
    };
}

declare_storage_value!(
    "Days since 1970-01-01 represented as a DuckDB `DATE`.",
    DateValue,
    i32,
    Date,
    Date,
    ffi::duckdb_v2_value_get_date,
    Hash,
    PartialOrd,
    Ord
);
declare_storage_value!(
    "Microseconds since midnight represented as a DuckDB `TIME`.",
    TimeValue,
    i64,
    Time,
    Time,
    ffi::duckdb_v2_value_get_time,
    Hash,
    PartialOrd,
    Ord
);
declare_storage_value!(
    "Nanoseconds since midnight represented as a DuckDB `TIME_NS`.",
    TimeNsValue,
    i64,
    TimeNs,
    TimeNs,
    ffi::duckdb_v2_value_get_time_ns,
    Hash,
    PartialOrd,
    Ord
);
declare_storage_value!(
    "Packed `TIME_TZ`: microseconds in the high 40 bits, reverse-ordered UTC offset in the low 24.",
    TimeTzValue,
    u64,
    TimeTz,
    TimeTz,
    ffi::duckdb_v2_value_get_time_tz
);
declare_storage_value!(
    "Microseconds since 1970-01-01 represented as a DuckDB `TIMESTAMP`.",
    TimestampValue,
    i64,
    Timestamp,
    Timestamp,
    ffi::duckdb_v2_value_get_timestamp,
    Hash,
    PartialOrd,
    Ord
);
declare_storage_value!(
    "Seconds since 1970-01-01 represented as a DuckDB `TIMESTAMP_SEC`.",
    TimestampSecValue,
    i64,
    TimestampSec,
    TimestampSec,
    ffi::duckdb_v2_value_get_timestamp_sec,
    Hash,
    PartialOrd,
    Ord
);
declare_storage_value!(
    "Milliseconds since 1970-01-01 represented as a DuckDB `TIMESTAMP_MS`.",
    TimestampMsValue,
    i64,
    TimestampMs,
    TimestampMs,
    ffi::duckdb_v2_value_get_timestamp_ms,
    Hash,
    PartialOrd,
    Ord
);
declare_storage_value!(
    "Nanoseconds since 1970-01-01 represented as a DuckDB `TIMESTAMP_NS`.",
    TimestampNsValue,
    i64,
    TimestampNs,
    TimestampNs,
    ffi::duckdb_v2_value_get_timestamp_ns,
    Hash,
    PartialOrd,
    Ord
);
declare_storage_value!(
    "UTC microseconds since 1970-01-01 represented as a DuckDB `TIMESTAMP_TZ`.",
    TimestampTzValue,
    i64,
    TimestampTz,
    TimestampTz,
    ffi::duckdb_v2_value_get_timestamp_tz,
    Hash,
    PartialOrd,
    Ord
);
declare_storage_value!(
    "UTC nanoseconds since 1970-01-01 represented as a DuckDB `TIMESTAMP_TZ_NS`.",
    TimestampTzNsValue,
    i64,
    TimestampTzNs,
    TimestampTzNs,
    ffi::duckdb_v2_value_get_timestamp_tz_ns,
    Hash,
    PartialOrd,
    Ord
);

impl TimeTzValue {
    /// Number of low bits reserved for the reversed UTC offset.
    pub const OFFSET_BITS: u32 = 24;
    /// Mask selecting the reversed UTC offset.
    pub const OFFSET_MASK: u64 = (1 << Self::OFFSET_BITS) - 1;
    /// Maximum UTC offset in seconds (15:59:59).
    pub const MAX_OFFSET: i32 = 16 * 60 * 60 - 1;

    /// Packs microseconds since midnight and a UTC offset in seconds, positive east, without validation.
    pub const fn new(micros: i64, offset_seconds: i32) -> Self {
        Self(((micros as u64) << Self::OFFSET_BITS) | ((Self::MAX_OFFSET - offset_seconds) as u64 & Self::OFFSET_MASK))
    }

    /// Microseconds since midnight.
    pub const fn micros(self) -> i64 {
        (self.0 >> Self::OFFSET_BITS) as i64
    }

    /// UTC offset in seconds, positive east.
    pub const fn offset_seconds(self) -> i32 {
        Self::MAX_OFFSET - (self.0 & Self::OFFSET_MASK) as i32
    }
}

/// A DuckDB `INTERVAL` split into month, day, and microsecond components.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntervalValue {
    /// Whole months.
    pub months: i32,
    /// Whole days.
    pub days: i32,
    /// Remaining microseconds.
    pub micros: i64,
}

impl DuckDBType for IntervalValue {
    fn logical_type<C: FFILink + ?Sized>(link: &C) -> Result<LogicalType> {
        link.logical_type_create_from_id(LogicalTypeID::Interval, Parameters::None)
    }
}

impl ToValue for IntervalValue {
    fn value<C: FFILink + ?Sized>(&self, link: &C) -> Result<Value> {
        link.create_value(ValueInput::Interval {
            months: self.months,
            days: self.days,
            micros: self.micros,
        })
    }
}

impl FromValue for IntervalValue {
    fn _get_inner(value: &Value) -> Result<Self> {
        check_api_call!(ffi::duckdb_v2_value_get_interval, value.raw(), RET).map(|raw| Self {
            months: raw.months,
            days: raw.days,
            micros: raw.micros,
        })
    }
}

declare_vector_element!(IntervalValue, Interval);

// SAFETY: every field combination is a valid `INTERVAL`.
unsafe impl WritableNativeElement for IntervalValue {
    type Write = Self;

    #[inline]
    unsafe fn write(
        vector: &mut crate::vector::VectorWriter<'_, Self>,
        index: usize,
        value: &Self::Write,
    ) -> Result<()> {
        // SAFETY: The caller guarantees a writable, flat, in-bounds row.
        unsafe { vector.write_internal(index, Some(*value)) }
    }
}

#[cfg(feature = "jiff")]
mod external {
    use ::jiff::{SignedDuration, Span, Timestamp, civil};

    use crate::types::{DateValue, IntervalValue, TimestampMsValue, TimestampTzValue, TimestampValue};
    use crate::types::{TimeNsValue, TimeValue, TimestampNsValue, TimestampSecValue, TimestampTzNsValue};

    const EPOCH_DATE: civil::Date = civil::date(1970, 1, 1);
    const EPOCH_DATETIME: civil::DateTime = civil::datetime(1970, 1, 1, 0, 0, 0, 0);

    /// Whether a raw timestamp is one of DuckDB's `infinity` / `-infinity` sentinels.
    const fn is_infinity(raw: i64) -> bool {
        raw == i64::MAX || raw == -i64::MAX
    }

    /// Returns `None` for infinity or dates outside jiff's range.
    impl From<DateValue> for Option<civil::Date> {
        fn from(value: DateValue) -> Self {
            EPOCH_DATE
                .checked_add(SignedDuration::from_hours(i64::from(value.0) * 24))
                .ok()
        }
    }

    impl From<civil::Date> for DateValue {
        fn from(value: civil::Date) -> Self {
            // jiff dates span years -9999..=9999, so the day count always fits in `i32`.
            Self((value.duration_since(EPOCH_DATE).as_secs() / 86_400) as i32)
        }
    }

    // jiff rejects results before midnight or at/after 24:00.
    fn time_from_nanos(nanos: i64) -> Option<civil::Time> {
        civil::Time::midnight()
            .checked_add(SignedDuration::from_nanos(nanos))
            .ok()
    }

    /// Returns `None` for times outside midnight (inclusive) to 24:00 (exclusive).
    impl From<TimeValue> for Option<civil::Time> {
        fn from(value: TimeValue) -> Self {
            time_from_nanos(value.0.checked_mul(1_000)?)
        }
    }

    /// Returns `None` for times outside midnight (inclusive) to 24:00 (exclusive).
    impl From<TimeNsValue> for Option<civil::Time> {
        fn from(value: TimeNsValue) -> Self {
            time_from_nanos(value.0)
        }
    }

    fn checked_timestamp_units(duration: SignedDuration, units_per_second: i64) -> crate::Result<i64> {
        let nanos_per_unit = 1_000_000_000 / units_per_second;
        // `subsec_nanos` has the same sign as `as_secs`, so both parts truncate toward zero together.
        let subsec_nanos = i64::from(duration.subsec_nanos());
        if subsec_nanos % nanos_per_unit != 0 {
            return Err(crate::Error::invalid_input(
                "timestamp precision cannot be represented exactly",
            ));
        }
        let units = duration
            .as_secs()
            .checked_mul(units_per_second)
            .and_then(|units| units.checked_add(subsec_nanos / nanos_per_unit))
            .ok_or_else(|| crate::Error::invalid_input("timestamp is out of range"))?;
        if is_infinity(units) {
            return Err(crate::Error::invalid_input("timestamp is reserved for infinity"));
        }
        Ok(units)
    }

    impl TryFrom<civil::Time> for TimeValue {
        type Error = crate::Error;

        fn try_from(value: civil::Time) -> crate::Result<Self> {
            checked_timestamp_units(value.duration_since(civil::Time::midnight()), 1_000_000).map(Self)
        }
    }

    impl TryFrom<civil::Time> for TimeNsValue {
        type Error = crate::Error;

        fn try_from(value: civil::Time) -> crate::Result<Self> {
            checked_timestamp_units(value.duration_since(civil::Time::midnight()), 1_000_000_000).map(Self)
        }
    }

    impl TryFrom<civil::DateTime> for TimestampValue {
        type Error = crate::Error;

        fn try_from(value: civil::DateTime) -> crate::Result<Self> {
            checked_timestamp_units(value.duration_since(EPOCH_DATETIME), 1_000_000).map(Self)
        }
    }

    impl TryFrom<civil::DateTime> for TimestampSecValue {
        type Error = crate::Error;

        fn try_from(value: civil::DateTime) -> crate::Result<Self> {
            checked_timestamp_units(value.duration_since(EPOCH_DATETIME), 1).map(Self)
        }
    }

    impl TryFrom<civil::DateTime> for TimestampMsValue {
        type Error = crate::Error;

        fn try_from(value: civil::DateTime) -> crate::Result<Self> {
            checked_timestamp_units(value.duration_since(EPOCH_DATETIME), 1_000).map(Self)
        }
    }

    impl TryFrom<civil::DateTime> for TimestampNsValue {
        type Error = crate::Error;

        fn try_from(value: civil::DateTime) -> crate::Result<Self> {
            checked_timestamp_units(value.duration_since(EPOCH_DATETIME), 1_000_000_000).map(Self)
        }
    }

    impl TryFrom<Timestamp> for TimestampTzValue {
        type Error = crate::Error;

        fn try_from(value: Timestamp) -> crate::Result<Self> {
            checked_timestamp_units(value.as_duration(), 1_000_000).map(Self)
        }
    }

    impl TryFrom<Timestamp> for TimestampTzNsValue {
        type Error = crate::Error;

        fn try_from(value: Timestamp) -> crate::Result<Self> {
            checked_timestamp_units(value.as_duration(), 1_000_000_000).map(Self)
        }
    }

    // jiff has no time-with-offset type.

    // A span keeps months and days as calendar units and the rest as exact time, like an interval,
    // so adding the converted span to a `civil::DateTime` matches DuckDB's `TIMESTAMP` arithmetic, and
    // adding it to a `Zoned` in the session's `TimeZone` matches its `TIMESTAMPTZ` arithmetic.
    /// Maps months, days, and micros to the span's months, days, and microseconds, without balancing.
    ///
    /// Returns `None` when the components have mixed signs (such as `1 month -1 day`), which a span
    /// cannot represent, or when a component exceeds jiff's limit for its unit (about 20,000 years).
    impl From<IntervalValue> for Option<Span> {
        fn from(value: IntervalValue) -> Self {
            let IntervalValue { months, days, micros } = value;
            let signs = [i64::from(months).signum(), i64::from(days).signum(), micros.signum()];
            if signs.contains(&1) && signs.contains(&-1) {
                return None;
            }
            Span::new()
                .try_months(months)
                .ok()?
                .try_days(days)
                .ok()?
                .try_microseconds(micros)
                .ok()
        }
    }

    /// Folds years into months, weeks into days, and every clock unit into micros.
    ///
    /// Fails when the nanoseconds are not a whole number of microseconds.
    impl TryFrom<Span> for IntervalValue {
        type Error = crate::Error;

        fn try_from(value: Span) -> crate::Result<Self> {
            let nanoseconds = value.get_nanoseconds();
            if nanoseconds % 1_000 != 0 {
                return Err(crate::Error::invalid_input(
                    "interval precision cannot be represented exactly",
                ));
            }
            let months = i128::from(value.get_years()) * 12 + i128::from(value.get_months());
            let days = i128::from(value.get_weeks()) * 7 + i128::from(value.get_days());
            let micros = i128::from(value.get_hours()) * 3_600_000_000
                + i128::from(value.get_minutes()) * 60_000_000
                + i128::from(value.get_seconds()) * 1_000_000
                + i128::from(value.get_milliseconds()) * 1_000
                + i128::from(value.get_microseconds())
                + i128::from(nanoseconds / 1_000);
            let out_of_range = |_| crate::Error::invalid_input("interval is out of range");
            Ok(Self {
                months: i32::try_from(months).map_err(out_of_range)?,
                days: i32::try_from(days).map_err(out_of_range)?,
                micros: i64::try_from(micros).map_err(out_of_range)?,
            })
        }
    }

    // Timestamps without a time zone map to `civil::DateTime`; only the TZ variants map to `Timestamp`.
    /// Returns `None` for infinity or timestamps outside jiff's range.
    impl From<TimestampValue> for Option<civil::DateTime> {
        fn from(value: TimestampValue) -> Self {
            EPOCH_DATETIME.checked_add(SignedDuration::from_micros(value.0)).ok()
        }
    }

    /// Returns `None` for infinity or timestamps outside jiff's range.
    impl From<TimestampSecValue> for Option<civil::DateTime> {
        fn from(value: TimestampSecValue) -> Self {
            EPOCH_DATETIME.checked_add(SignedDuration::from_secs(value.0)).ok()
        }
    }

    /// Returns `None` for infinity or timestamps outside jiff's range.
    impl From<TimestampMsValue> for Option<civil::DateTime> {
        fn from(value: TimestampMsValue) -> Self {
            EPOCH_DATETIME.checked_add(SignedDuration::from_millis(value.0)).ok()
        }
    }

    /// Returns `None` for infinity or timestamps outside jiff's range.
    impl From<TimestampTzValue> for Option<Timestamp> {
        fn from(value: TimestampTzValue) -> Self {
            Timestamp::from_microsecond(value.0).ok()
        }
    }

    fn finite_timestamp_nanos(value: i64) -> Option<SignedDuration> {
        if is_infinity(value) {
            return None;
        }
        Some(SignedDuration::from_nanos(value))
    }

    /// Returns `None` for infinity; every finite nanosecond timestamp fits jiff's range.
    impl From<TimestampNsValue> for Option<civil::DateTime> {
        fn from(value: TimestampNsValue) -> Self {
            EPOCH_DATETIME.checked_add(finite_timestamp_nanos(value.0)?).ok()
        }
    }

    /// Returns `None` for infinity; every finite nanosecond timestamp fits jiff's range.
    impl From<TimestampTzNsValue> for Option<Timestamp> {
        fn from(value: TimestampTzNsValue) -> Self {
            Timestamp::from_duration(finite_timestamp_nanos(value.0)?).ok()
        }
    }

    #[cfg(test)]
    mod tests {
        use ::jiff::{Timestamp, civil, tz::Offset};

        use crate::{
            Parameters,
            environment::{Environment, StorageLocation},
            types::{
                DateValue, TimeNsValue, TimeValue, TimestampMsValue, TimestampNsValue, TimestampSecValue,
                TimestampTzNsValue, TimestampTzValue, TimestampValue,
            },
        };

        fn naive(s: &str) -> civil::DateTime {
            s.parse().unwrap()
        }

        fn utc(value: civil::DateTime) -> Timestamp {
            Offset::UTC.to_timestamp(value).unwrap()
        }

        #[test]
        fn test_checked_jiff_to_storage() -> crate::Result<()> {
            for text in ["1992-09-20 11:30:00", "1960-01-02 03:04:05"] {
                let value = naive(text);
                assert_eq!(
                    Option::<civil::DateTime>::from(TimestampValue::try_from(value)?),
                    Some(value)
                );
                assert_eq!(
                    Option::<civil::DateTime>::from(TimestampSecValue::try_from(value)?),
                    Some(value)
                );
                assert_eq!(
                    Option::<civil::DateTime>::from(TimestampMsValue::try_from(value)?),
                    Some(value)
                );
                assert_eq!(
                    Option::<civil::DateTime>::from(TimestampNsValue::try_from(value)?),
                    Some(value)
                );
                let utc = utc(value);
                assert_eq!(Option::<Timestamp>::from(TimestampTzValue::try_from(utc)?), Some(utc));
                assert_eq!(Option::<Timestamp>::from(TimestampTzNsValue::try_from(utc)?), Some(utc));
            }
            let micros = naive("1960-01-02 03:04:05.123456");
            assert_eq!(
                Option::<civil::DateTime>::from(TimestampValue::try_from(micros)?),
                Some(micros)
            );
            let millis = naive("1960-01-02 03:04:05.123");
            assert_eq!(
                Option::<civil::DateTime>::from(TimestampMsValue::try_from(millis)?),
                Some(millis)
            );
            let nanos = naive("1960-01-02 03:04:05.123456789");
            assert_eq!(
                Option::<civil::DateTime>::from(TimestampNsValue::try_from(nanos)?),
                Some(nanos)
            );
            assert_eq!(
                Option::<civil::Time>::from(TimeValue::try_from(micros.time())?),
                Some(micros.time())
            );
            assert_eq!(
                Option::<civil::Time>::from(TimeNsValue::try_from(nanos.time())?),
                Some(nanos.time())
            );
            Ok(())
        }

        #[test]
        fn test_checked_jiff_to_storage_rejects_loss_and_overflow() {
            let nanos = naive("1992-09-20 11:30:00.123456789");
            assert!(TimestampSecValue::try_from(nanos).is_err());
            assert!(TimestampMsValue::try_from(nanos).is_err());
            assert!(TimestampValue::try_from(nanos).is_err());
            assert!(TimestampTzValue::try_from(utc(nanos)).is_err());
            assert!(TimeValue::try_from(nanos.time()).is_err());

            for value in [civil::DateTime::MIN, civil::DateTime::MAX] {
                assert!(TimestampNsValue::try_from(value).is_err());
            }
            for value in [Timestamp::MIN, Timestamp::MAX] {
                assert!(TimestampTzNsValue::try_from(value).is_err());
            }
            for raw in [i64::MAX, -i64::MAX] {
                let value = Timestamp::from_nanosecond(i128::from(raw)).unwrap();
                assert!(TimestampNsValue::try_from(Offset::UTC.to_datetime(value)).is_err());
                assert!(TimestampTzNsValue::try_from(value).is_err());
            }
        }

        #[test]
        fn test_date_epoch_round_trip() {
            for (year, month, day, days) in [(1969, 12, 31, -1), (1970, 1, 1, 0), (1992, 9, 20, 8298)] {
                let date = civil::date(year, month, day);
                let raw = DateValue::from(date);
                assert_eq!(raw.0, days);
                assert_eq!(Option::<civil::Date>::from(raw), Some(date));
            }
        }

        #[test]
        fn test_date() -> crate::Result<()> {
            let env = Environment::new()?;
            let instance = env.open(StorageLocation::InMemory)?;
            let conn = instance.connect()?;

            let mut result = conn.query("SELECT DATE '1992-09-20';", Parameters::None)?;

            let chunk = result.next_chunk()?.unwrap();
            let vec = chunk.get_vector_at::<DateValue>(0)?;

            assert_eq!(vec.len(), 1);
            let date = Option::<civil::Date>::from(*vec.reader()?.get(0)?.unwrap()).unwrap();

            assert_eq!(date, civil::date(1992, 9, 20));

            Ok(())
        }

        #[test]
        fn test_timestamp_conversion() -> crate::Result<()> {
            let env = Environment::new()?;
            let instance = env.open(StorageLocation::InMemory)?;
            let conn = instance.connect()?;
            {
                let mut result =
                    conn.query("SELECT TIMESTAMP_NS '1992-09-20 11:30:00.123456789';", Parameters::None)?;
                let chunk = result.next_chunk()?.unwrap();
                let vec = chunk.get_vector_at::<TimestampNsValue>(0)?;
                let timestamp = Option::<civil::DateTime>::from(*vec.reader()?.get(0)?.unwrap()).unwrap();
                assert_eq!(timestamp, naive("1992-09-20 11:30:00.123456789"));
            }
            {
                let mut result = conn.query("SELECT TIMESTAMP '1992-09-20 11:30:00.123456789';", Parameters::None)?;
                let chunk = result.next_chunk()?.unwrap();
                let vec = chunk.get_vector_at::<TimestampValue>(0)?;
                let val = *(vec.reader()?.get(0)?.unwrap());
                let timestamp = Option::<civil::DateTime>::from(val).unwrap();
                assert_eq!(timestamp, naive("1992-09-20 11:30:00.123456"));
            }
            {
                let mut result =
                    conn.query("SELECT TIMESTAMP_MS '1992-09-20 11:30:00.123456789';", Parameters::None)?;
                let chunk = result.next_chunk()?.unwrap();
                let vec = chunk.get_vector_at::<TimestampMsValue>(0)?;
                let val = *(vec.reader()?.get(0)?.unwrap());
                let timestamp = Option::<civil::DateTime>::from(val).unwrap();
                assert_eq!(timestamp, naive("1992-09-20 11:30:00.123"));
            }
            {
                let mut result = conn.query("SELECT TIMESTAMP_S '1992-09-20 11:30:00.123456789';", Parameters::None)?;
                let chunk = result.next_chunk()?.unwrap();
                let vec = chunk.get_vector_at::<TimestampSecValue>(0)?;
                let val = *(vec.reader()?.get(0)?.unwrap());
                let timestamp = Option::<civil::DateTime>::from(val).unwrap();
                assert_eq!(timestamp, naive("1992-09-20 11:30:00"));
            }
            {
                let mut result = conn.query(
                    "SELECT TIMESTAMPTZ '1992-09-20 11:30:00.123456789+00:00';",
                    Parameters::None,
                )?;
                let chunk = result.next_chunk()?.unwrap();
                let vec = chunk.get_vector_at::<TimestampTzValue>(0)?;
                let val = *(vec.reader()?.get(0)?.unwrap());
                let timestamp = Option::<Timestamp>::from(val).unwrap();
                assert_eq!(timestamp, "1992-09-20T11:30:00.123456Z".parse::<Timestamp>().unwrap());
            }
            {
                let mut result = conn.query(
                    "SELECT TIMESTAMPTZ '1992-09-20 12:30:00.123456789+01:00';",
                    Parameters::None,
                )?;
                let chunk = result.next_chunk()?.unwrap();
                let vec = chunk.get_vector_at::<TimestampTzValue>(0)?;
                let val = *(vec.reader()?.get(0)?.unwrap());
                let timestamp = Option::<Timestamp>::from(val).unwrap();
                assert_eq!(
                    timestamp,
                    "1992-09-20T12:30:00.123456+01:00".parse::<Timestamp>().unwrap()
                );
            }
            {
                let mut result = conn.query(
                    "SELECT TIMESTAMPTZ_NS '1992-09-20 11:30:00.123456789+00:00';",
                    Parameters::None,
                )?;
                let chunk = result.next_chunk()?.unwrap();
                let vector = chunk.get_vector_at::<TimestampTzNsValue>(0)?;
                let timestamp = Option::<Timestamp>::from(*vector.reader()?.get(0)?.unwrap()).unwrap();
                assert_eq!(timestamp, utc(naive("1992-09-20 11:30:00.123456789")));
            }
            Ok(())
        }

        #[test]
        fn test_temporal_infinity_conversion() -> crate::Result<()> {
            let env = Environment::new()?;
            let instance = env.open(StorageLocation::InMemory)?;
            let conn = instance.connect()?;
            let mut result = conn.query(
                "SELECT
                'infinity'::TIMESTAMP_NS,
                '-infinity'::TIMESTAMP_NS,
                'infinity'::TIMESTAMPTZ_NS,
                '-infinity'::TIMESTAMPTZ_NS,
                'infinity'::DATE,
                '-infinity'::DATE,
                'infinity'::TIMESTAMP,
                '-infinity'::TIMESTAMP,
                'infinity'::TIMESTAMP_S,
                '-infinity'::TIMESTAMP_S,
                'infinity'::TIMESTAMP_MS,
                '-infinity'::TIMESTAMP_MS,
                'infinity'::TIMESTAMPTZ,
                '-infinity'::TIMESTAMPTZ;",
                Parameters::None,
            )?;
            let chunk = result.next_chunk()?.unwrap();

            for index in 0..2 {
                let vector = chunk.get_vector_at::<TimestampNsValue>(index)?;
                assert!(Option::<civil::DateTime>::from(*vector.reader()?.get(0)?.unwrap()).is_none());
            }
            for index in 2..4 {
                let vector = chunk.get_vector_at::<TimestampTzNsValue>(index)?;
                assert!(Option::<Timestamp>::from(*vector.reader()?.get(0)?.unwrap()).is_none());
            }
            for index in 4..6 {
                let vector = chunk.get_vector_at::<DateValue>(index)?;
                assert!(Option::<civil::Date>::from(*vector.reader()?.get(0)?.unwrap()).is_none());
            }
            for index in 6..8 {
                let vector = chunk.get_vector_at::<TimestampValue>(index)?;
                assert!(Option::<civil::DateTime>::from(*vector.reader()?.get(0)?.unwrap()).is_none());
            }
            for index in 8..10 {
                let vector = chunk.get_vector_at::<TimestampSecValue>(index)?;
                assert!(Option::<civil::DateTime>::from(*vector.reader()?.get(0)?.unwrap()).is_none());
            }
            for index in 10..12 {
                let vector = chunk.get_vector_at::<TimestampMsValue>(index)?;
                assert!(Option::<civil::DateTime>::from(*vector.reader()?.get(0)?.unwrap()).is_none());
            }
            for index in 12..14 {
                let vector = chunk.get_vector_at::<TimestampTzValue>(index)?;
                assert!(Option::<Timestamp>::from(*vector.reader()?.get(0)?.unwrap()).is_none());
            }

            Ok(())
        }

        #[test]
        fn test_time_out_of_range_conversion() {
            let wrapping_micros = ((1_i64 << 32) + 5) * 1_000_000;
            for raw in [-1, i64::MIN, i64::MAX, wrapping_micros, 86_400_000_000] {
                assert!(Option::<civil::Time>::from(TimeValue(raw)).is_none());
            }
            for raw in [-1, i64::MIN, i64::MAX, wrapping_micros * 1_000, 86_400_000_000_000] {
                assert!(Option::<civil::Time>::from(TimeNsValue(raw)).is_none());
            }
        }

        #[test]
        fn test_time_conversion() -> crate::Result<()> {
            let env = Environment::new()?;
            let instance = env.open(StorageLocation::InMemory)?;
            let conn = instance.connect()?;

            {
                let mut result = conn.query("SELECT TIME '1992-09-20 11:30:00.123456';", Parameters::None)?;
                let chunk = result.next_chunk()?.unwrap();
                let vec = chunk.get_vector_at::<TimeValue>(0)?;
                let val = *(vec.reader()?.get(0)?.unwrap());
                let time = Option::<civil::Time>::from(val).unwrap();
                assert_eq!(time, "11:30:00.123456".parse::<civil::Time>().unwrap());
            }
            {
                let mut result = conn.query("SELECT '15:30:00.123456789'::TIME_NS;", Parameters::None)?;
                let chunk = result.next_chunk()?.unwrap();
                let vec = chunk.get_vector_at::<TimeNsValue>(0)?;
                let val = *(vec.reader()?.get(0)?.unwrap());
                let time = Option::<civil::Time>::from(val).unwrap();
                assert_eq!(time, "15:30:00.123456789".parse::<civil::Time>().unwrap());
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Parameters,
        environment::{Environment, StorageLocation},
    };

    #[test]
    fn test_temporal_ordering_and_hashing() {
        fn check<T: Copy + std::fmt::Debug + Ord + std::hash::Hash>(negative: T, zero: T, positive: T) {
            let ordered = std::collections::BTreeSet::from([positive, negative, zero, zero]);
            assert_eq!(ordered.into_iter().collect::<Vec<_>>(), [negative, zero, positive]);
            let hashed = std::collections::HashSet::from([negative, zero, zero, positive]);
            assert_eq!(hashed.len(), 3);
            assert!(hashed.contains(&zero));
        }

        check(super::DateValue(-1), super::DateValue(0), super::DateValue(1));
        check(super::TimeValue(-1), super::TimeValue(0), super::TimeValue(1));
        check(super::TimeNsValue(-1), super::TimeNsValue(0), super::TimeNsValue(1));
        check(
            super::TimestampValue(-1),
            super::TimestampValue(0),
            super::TimestampValue(1),
        );
        check(
            super::TimestampSecValue(-1),
            super::TimestampSecValue(0),
            super::TimestampSecValue(1),
        );
        check(
            super::TimestampMsValue(-1),
            super::TimestampMsValue(0),
            super::TimestampMsValue(1),
        );
        check(
            super::TimestampNsValue(-1),
            super::TimestampNsValue(0),
            super::TimestampNsValue(1),
        );
        check(
            super::TimestampTzValue(-1),
            super::TimestampTzValue(0),
            super::TimestampTzValue(1),
        );
        check(
            super::TimestampTzNsValue(-1),
            super::TimestampTzNsValue(0),
            super::TimestampTzNsValue(1),
        );
    }

    #[test]
    fn test_time_tz_packing() -> crate::Result<()> {
        let env = Environment::new()?;
        let instance = env.open(StorageLocation::InMemory)?;
        let conn = instance.connect()?;
        let mut result = conn.query("SELECT TIMETZ '12:34:56+01:30';", Parameters::None)?;
        let chunk = result.next_chunk()?.unwrap();
        let vector = chunk.get_vector_at::<super::TimeTzValue>(0)?;
        let value = *vector.reader()?.get(0)?.unwrap();
        let micros = (12 * 3600 + 34 * 60 + 56) * 1_000_000;
        assert_eq!(value.micros(), micros);
        assert_eq!(value.offset_seconds(), 5400);
        assert_eq!(value, super::TimeTzValue::new(micros, 5400));

        for micros in [0, 1, 86_400_000_000] {
            for offset in [
                -super::TimeTzValue::MAX_OFFSET,
                -5400,
                0,
                5400,
                super::TimeTzValue::MAX_OFFSET,
            ] {
                let value = super::TimeTzValue::new(micros, offset);
                assert_eq!(value.micros(), micros);
                assert_eq!(value.offset_seconds(), offset);
            }
        }
        Ok(())
    }

    #[test]
    fn test_interval_conversion() -> crate::Result<()> {
        let env = Environment::new()?;
        let instance = env.open(StorageLocation::InMemory)?;
        let conn = instance.connect()?;

        let mut result = conn.query(
            "SELECT INTERVAL '1 year 2 months 3 days 4 hours 5 minutes 6 seconds';",
            Parameters::None,
        )?;
        let chunk = result.next_chunk()?.unwrap();
        let vec = chunk.get_vector_at::<super::IntervalValue>(0)?;
        let val = *(vec.reader()?.get(0)?.unwrap());

        assert_eq!(val.months, 14);
        assert_eq!(val.days, 3);
        assert_eq!(val.micros, 14706000000);

        Ok(())
    }
}

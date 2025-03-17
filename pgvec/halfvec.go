package pgvec

import (
	"database/sql/driver"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5/pgtype"
	"github.com/x448/float16"
)

type HalfVector struct {
	vals []float16.Float16
}

// NewFromFloat32 creates a new HalfVector from a slice of float32 values
func NewFromFloat32(values []float32) *HalfVector {
	buf := make([]float16.Float16, len(values))
	for i, v := range values {
		buf[i] = float16.Fromfloat32(v)
	}

	return &HalfVector{vals: buf}
}

func bytes16ToFloat32(b []byte) float32 {
	bits := binary.BigEndian.Uint16(b)
	return float16.Frombits(bits).Float32()
}

// ToFloat32 converts the HalfVector back to float32 values
func (hv *HalfVector) ToFloat32() []float32 {
	floatValues := make([]float32, len(hv.vals))
	for i, v := range hv.vals {
		floatValues[i] = float16.Float16(v).Float32()
	}
	return floatValues
}

func (hv *HalfVector) EncodeBinary() ([]byte, error) {
	buf := make([]byte, 4+(2*len(hv.vals)))

	binary.BigEndian.PutUint16(buf[:2], uint16(len(hv.vals)))
	for i, v := range hv.vals {
		start := 4 + (2 * i)
		binary.BigEndian.PutUint16(buf[start:start+2], uint16(v))
	}
	return buf, nil
}

func (hv *HalfVector) Scan(value interface{}) error {
	valueBytes, ok := value.([]byte)
	if !ok {
		return errors.New("unable to convert value to bytes")
	}

	ohv, err := DecodeBinary(valueBytes)
	if err != nil {
		return err
	}

	hv.vals = ohv.vals
	return nil
}

// DecodeBinary decodes a binary representation back to a HalfVector
func DecodeBinary(data []byte) (*HalfVector, error) {
	dimBytes := data[:2]
	body := data[4:]

	length := binary.BigEndian.Uint16(dimBytes)

	if length < 0 {
		return nil, fmt.Errorf("invalid vector length: %d", length)
	}

	if len(body) != 2*int(length) {
		return nil, fmt.Errorf("body must be 2 bytes per item")
	}

	vals := make([]float16.Float16, length)

	for i := 0; i < int(length); i++ {
		start := 4 + (i * 2)
		vals[i] = float16.Float16(binary.BigEndian.Uint16(data[start : start+2]))
	}

	return &HalfVector{vals: vals}, nil
}

type VectorCodec struct{}

func (c *VectorCodec) FormatSupported(format int16) bool {
	return format == pgtype.BinaryFormatCode
}

func (c *VectorCodec) PreferredFormat() int16 {
	return pgtype.BinaryFormatCode
}

func (c *VectorCodec) DecodeValue(m *pgtype.Map, oid uint32, format int16, src []byte) (interface{}, error) {
	if src == nil {
		return nil, nil
	}

	if format != pgtype.BinaryFormatCode {
		return nil, fmt.Errorf("unsupported format code: %d", format)
	}

	// Decode the binary data into our HalfVector type
	halfVec, err := DecodeBinary(src)
	if err != nil {
		return nil, err
	}

	return halfVec, nil
}

type VectorEncodePlan struct {
	target any
}

func (p *VectorEncodePlan) Encode(value any, buf []byte) ([]byte, error) {
	var v any
	if value != nil {
		v = value
	} else {
		v = p.target
	}

	switch val := v.(type) {
	case *HalfVector:
		encoded, err := val.EncodeBinary()
		if err != nil {
			return nil, err
		}
		return append(buf, encoded...), nil
	case HalfVector:
		encoded, err := (&val).EncodeBinary()
		if err != nil {
			return nil, err
		}
		return append(buf, encoded...), nil
	default:
		return nil, fmt.Errorf("unsupported source type %T", v)
	}
}

func (c *VectorCodec) PlanEncode(m *pgtype.Map, oid uint32, format int16, value any) pgtype.EncodePlan {
	if format != pgtype.BinaryFormatCode {
		return nil
	}

	return &VectorEncodePlan{target: value}
}

// VectorScanPlan implements pgtype.ScanPlan
type VectorScanPlan struct {
	Target any
}

func (p *VectorScanPlan) Scan(src []byte, dst any) error {
	if src == nil {
		return nil
	}

	maxBytes := 10
	if len(src) < maxBytes {
		maxBytes = len(src)
	}

	halfVec, err := DecodeBinary(src)
	if err != nil {
		return fmt.Errorf("failed to decode binary data: %w", err)
	}

	switch v := dst.(type) {
	case *HalfVector:
		*v = *halfVec
		return nil
	default:
		return fmt.Errorf("unsupported target type %T", dst)
	}
}

func (c *VectorCodec) PlanScan(m *pgtype.Map, oid uint32, format int16, target any) pgtype.ScanPlan {

	if format != pgtype.BinaryFormatCode && format != pgtype.TextFormatCode {
		return nil
	}

	// Return a plan based on the target type
	switch target.(type) {
	case *[]float32, *HalfVector:
		return &VectorScanPlan{Target: target}
	default:
		return nil
	}
}

func (c *VectorCodec) DecodeDatabaseSQLValue(m *pgtype.Map, oid uint32, format int16, src []byte) (driver.Value, error) {
	if src == nil {
		return nil, nil
	}

	if format != pgtype.BinaryFormatCode {
		return nil, fmt.Errorf("unsupported format code: %d", format)
	}

	// Decode the binary data into our HalfVector type
	halfVec, err := DecodeBinary(src)
	if err != nil {
		return nil, err
	}

	return halfVec.ToFloat32(), nil
}

func (c *VectorCodec) EncodeDatabaseSQLValue(m *pgtype.Map, oid uint32, format int16, src driver.Value) ([]byte, error) {
	if src == nil {
		return nil, nil
	}

	if format != pgtype.BinaryFormatCode {
		return nil, fmt.Errorf("unsupported format code: %d", format)
	}

	switch v := src.(type) {
	case *HalfVector:
		return v.EncodeBinary()
	case HalfVector:
		return v.EncodeBinary()
	default:
		return nil, fmt.Errorf("unsupported source type %T", src)
	}
}

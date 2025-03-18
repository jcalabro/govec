package pgvec

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"

	govec "github.com/whyrusleeping/govec"
)

func normalDotProduct(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}

	return sum
}

func TestLoadVector(t *testing.T) {
	config, err := pgxpool.ParseConfig(os.Getenv("TEST_DATABASE_URL"))
	if err != nil {
		t.Fatal("Unable to parse pool config: ", err)
	}

	ctx := context.TODO()

	vectorOID := uint32(616049)
	config.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		conn.TypeMap().RegisterType(&pgtype.Type{
			Name:  "halfvec",
			OID:   vectorOID,
			Codec: &VectorCodec{},
		})
		return nil
	}

	pgxpool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}

	_, _ = pgxpool.Exec(ctx, "DROP TABLE embeds")
	_, _ = pgxpool.Exec(ctx, "CREATE EXTENSION vector")
	_, err = pgxpool.Exec(ctx, `CREATE TABLE embeds (
    				id SERIAL PRIMARY KEY,
    				embed halfvec(512)
			)`)
	if err != nil {
		t.Fatal(err)
	}

	rows, err := pgxpool.Query(ctx, "select embed from embeds where id = 1")
	if err != nil {
		t.Fatal(err)
	}

	defer rows.Close()

	if !rows.Next() {
		t.Fatal("expected a value")
	}

	var out HalfVector
	if err := rows.Scan(&out); err != nil {
		t.Fatal(err)
	}

	fmt.Println(out.ToFloat32())

	fres := govec.DotProductFastFP16(out.vals, out.vals)
	sres := normalDotProduct(out.ToFloat32(), out.ToFloat32())

	fmt.Println(fres, sres)

	_, _ = pgxpool.Exec(ctx, "DROP TABLE embeds")
}

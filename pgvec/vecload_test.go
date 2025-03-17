package pgvec

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
)

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

	rows, err := pgxpool.Query(ctx, "select embedding from post_embeddings where post = 1184219013")
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
}

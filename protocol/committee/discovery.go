package committee

import (
	"context"
	"log"
	"os"

	peerpb "github.com/Lekssays/formica/protocol/proto/peer"
	redis "github.com/go-redis/redis/v8"
	"github.com/golang/protobuf/proto"
)

func SavePeer(peer peerpb.Peer) (bool, error) {
	var ctx = context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     os.Getenv("REDIS_ENDPOINT"),
		Password: "",
		DB:       0,
	})

	peerString := proto.MarshalTextString(&peer)

	err := rdb.SAdd(ctx, "peers", peerString).Err()
	if err != nil {
		return false, err
	}

	return true, nil
}

func GetPeers() ([]peerpb.Peer, error) {
	var ctx = context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     os.Getenv("REDIS_ENDPOINT"),
		Password: "",
		DB:       0,
	})

	peersStrings, err := rdb.SMembers(ctx, "peers").Result()
	if err != nil {
		return []peerpb.Peer{}, err
	}

	var peers []peerpb.Peer
	for i := 0; i < len(peersStrings); i++ {
		var peer peerpb.Peer
		err = proto.UnmarshalText(peersStrings[i], &peer)
		if err != nil {
			log.Println(err.Error())
		}
		peers = append(peers, peer)
	}

	return peers, nil
}

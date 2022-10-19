package formica

import (
	"time"

	"github.com/iotaledger/hive.go/core/generics/event"
)

// Events define events occurring within a formica payload.
type Events struct {
	BlockReceived *event.Event[*BlockReceivedEvent]
}

// newEvents returns a new Events object.
func newEvents() (new *Events) {
	return &Events{
		BlockReceived: event.New[*BlockReceivedEvent](),
	}
}

// Event defines the information passed when a formica event fires.
type BlockReceivedEvent struct {
	Purpose   uint32
	Data      string
	Block     string
	Timestamp time.Time
	BlockID   string
}

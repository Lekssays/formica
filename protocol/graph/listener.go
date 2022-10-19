package graph

import (
	"encoding/json"
	"flag"
	"log"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

const (
	GOSHIMMER_WEBSOCKETS_ENDPOINT = "0.0.0.0:8081"
)

type Data struct {
	ID          string `json:"id,omitempty"`
	Value       int    `json:"value,omitempty"`
	PayloadType int    `json:"payload_type,omitempty"`
}

type GHResponse struct {
	Type string `json:"type,omitempty"`
	Data Data   `json:"data,omitempty"`
}

var addr = flag.String("addr", GOSHIMMER_WEBSOCKETS_ENDPOINT, "http service address")

func RunLiveFeed(wg *sync.WaitGroup) {
	flag.Parse()
	log.SetFlags(0)

	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt)

	u := url.URL{Scheme: "ws", Host: *addr, Path: "/ws"}
	log.Printf("connecting to %s", u.String())

	c, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	if err != nil {
		log.Fatal("dial:", err)
	}
	defer c.Close()

	done := make(chan struct{})

	go func() {
		defer close(done)
		for {
			_, message, err := c.ReadMessage()
			if err != nil {
				log.Println("read:", err)
				return
			}

			var response GHResponse
			json.Unmarshal(message, &response)

			// PayloadType for Formica is 787 (see plugins/formica/formica.go)
			if strings.Contains(string(message), "\"payload_type\":787") {
				messageID := response.Data.ID
				log.Printf("MessageID: %s", messageID)
				mupdate, err := GetModelUpdate(string(messageID))
				if err != nil {
					log.Println("Error GetModelUpdate-", err.Error())
					continue
				}
				err = SaveModelUpdate(string(messageID), mupdate)
				if err != nil {
					log.Println("Error SaveModelUpdate-", err.Error())
				}
			}
		}
	}()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return
		case t := <-ticker.C:
			err := c.WriteMessage(websocket.TextMessage, []byte(t.String()))
			if err != nil {
				log.Println("write:", err)
				return
			}
		case <-interrupt:
			log.Println("interrupt")

			// Cleanly close the connection by sending a close message and then
			// waiting (with timeout) for the server to close the connection.
			err := c.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
			if err != nil {
				log.Println("write close:", err)
				return
			}
			select {
			case <-done:
			case <-time.After(time.Second):
			}
			return
		}
	}
}

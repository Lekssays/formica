package formica

import (
	"net/http"

	"github.com/iotaledger/goshimmer/packages/app/jsonmodels"
	"github.com/labstack/echo"
)

const (
	maxDataLength = 4096
)

func configureWebAPI() {
	deps.Server.POST("formica", SendFormicaBlock)
}

// SendFormicaBlock sends a formica message.
func SendFormicaBlock(c echo.Context) error {
	req := &Request{}
	if err := c.Bind(req); err != nil {
		return c.JSON(http.StatusBadRequest, jsonmodels.NewErrorResponse(err))
	}

	if len(req.Data) > maxDataLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "Data is too long"})
	}

	formicaPayload := NewPayload(req.Purpose, req.Data, req.Block)
	blk, err := deps.Tangle.IssuePayload(formicaPayload)
	if err != nil {
		return c.JSON(http.StatusBadRequest, Response{Error: err.Error()})
	}

	return c.JSON(http.StatusOK, Response{BlockID: blk.ID().Base58()})
}

// Request defines the formica block to send.
type Request struct {
	Purpose uint32 `json:"purpose"`
	Data    string `json:"data"`
	Block   string `json:"block"`
}

// Response contains the ID of the block sent.
type Response struct {
	BlockID string `json:"blockID,omitempty"`
	Error   string `json:"error,omitempty"`
}

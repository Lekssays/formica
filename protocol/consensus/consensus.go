package consensus

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Lekssays/formica/protocol/graph"
	"github.com/Lekssays/formica/protocol/plugins/formica"
	mupb "github.com/Lekssays/formica/protocol/proto/modelUpdate"
	scpb "github.com/Lekssays/formica/protocol/proto/score"
	"github.com/go-redis/redis/v8"
	"github.com/golang/protobuf/proto"
	"github.com/iotaledger/goshimmer/client"
	shell "github.com/ipfs/go-ipfs-api"
	"github.com/sbinet/npyio"
	"gonum.org/v1/gonum/mat"
)

const (
	INF                   = 1000
	THRESHOLD             = 0.1
	DECAY_RATE            = 0.0001
	DELTA                 = 0.01
	ALPHA                 = 0.5
	K                     = 10
	TRUST_PURPOSE_ID      = 21
	SIMILARITY_PURPOSE_ID = 22
	ALIGNMENT_PURPOSE_ID  = 23
	GRADIENTS_PURPOSE_ID  = 24
	PHI_PURPOSE_ID        = 25
	IPFS_ENDPOINT         = "http://193.206.183.35:5001"
)

type Payload struct {
	File []byte `json:"file"`
}

type Gradients struct {
	Content map[string][][]float64
}

// ComputeCS returns the cosine similarity of two vectors
func ComputeCS(a []float64, b []float64) float64 {
	count := 0
	length_a := len(a)
	length_b := len(b)
	if length_a > length_b {
		count = length_a
	} else {
		count = length_b
	}
	sumA := 0.0
	s1 := 0.0
	s2 := 0.0
	for k := 0; k < count; k++ {
		if k >= length_a {
			s2 += math.Pow(b[k], 2)
			continue
		}
		if k >= length_b {
			s1 += math.Pow(a[k], 2)
			continue
		}
		sumA += a[k] * b[k]
		s1 += math.Pow(a[k], 2)
		s2 += math.Pow(b[k], 2)
	}
	if s1 == 0 || s2 == 0 {
		return float64(0.0)
	}

	cosine := sumA / (math.Sqrt(math.Abs(s1)) * math.Sqrt(math.Abs(s2)))
	return cosine
}

func getAverage(a []float64) float64 {
	sum := 0.0
	for i := 0; i < len(a); i++ {
		sum += a[i]
	}
	return (float64(sum) / float64(len(a)))
}

func getMax(a []float64) float64 {
	max := -1.0
	for i := 0; i < len(a); i++ {
		if a[i] >= max {
			max = a[i]
		}
	}
	return max
}

func getMin(a []float64) float64 {
	min := 1000.0
	for i := 0; i < len(a); i++ {
		if a[i] <= min {
			min = a[i]
		}
	}
	return min
}

func ComputeCSMatrix(modelID string) ([][]float64, error) {
	clients, err := graph.GetClients(modelID)
	if err != nil {
		return [][]float64{}, err
	}

	csMatrix := make([][]float64, len(clients))
	for i := range csMatrix {
		csMatrix[i] = make([]float64, len(clients))
	}

	gradients, err := GetLatestGradients(modelID)
	if err != nil {
		return [][]float64{}, err
	}

	computed := make(map[string]bool)
	for i := 0; i < len(clients); i++ {
		for j := 0; j < len(clients); j++ {
			key := fmt.Sprintf("%d-%d", j, i)
			if !computed[key] {
				if i == j {
					csMatrix[i][j] = 0.0
				} else {
					csMatrix[i][j] = ComputeClassCS(gradients[clients[i]], gradients[clients[j]])
				}
				csMatrix[j][i] = csMatrix[i][j]
				computed[key] = true
			}
		}
	}
	return csMatrix, nil
}

func ComputeClassCS(a [][]float64, b [][]float64) float64 {
	cs := make([]float64, 0)
	for i := 0; i < len(a); i++ {
		rcs := ComputeCS(a[i], b[i])
		if rcs > 1.0 {
			rcs = 1.0
		}
		cs = append(cs, rcs)
	}
	return getAverage(cs)
}

func ComputeAlignmentScore(modelID string, csMatrix [][]float64) []float64 {
	clients, err := graph.GetClients(modelID)
	if err != nil {
		return []float64{}
	}

	algnScore := make([]float64, len(clients))
	for i := 0; i < len(clients); i++ {
		algnScore[i] = getMax(csMatrix[i])
	}
	return algnScore
}

func EvaluatePardoning(modelID string, algnScore []float64, csMatrix [][]float64) ([][]float64, []float64, error) {
	clients, err := graph.GetClients(modelID)
	if err != nil {
		return [][]float64{}, []float64{}, err
	}

	for i := 0; i < len(clients); i++ {
		for j := 0; j < len(clients); j++ {
			if algnScore[i] > algnScore[j] {
				csMatrix[i][j] *= math.Min(1, algnScore[i]/algnScore[j])
				csMatrix[j][i] = csMatrix[i][j]
			}
		}
		algnScore[i] = getMax(csMatrix[i])
	}
	return csMatrix, algnScore, nil
}

func ComputeTrust(modelID string, algnScore []float64) (map[string]float32, error) {
	trustScores, err := GetLatestTrustScores(modelID)
	if err != nil {
		fmt.Println("trustScores", err.Error())
		return map[string]float32{}, err
	}

	clients, err := graph.GetClients(modelID)
	if err != nil {
		return map[string]float32{}, err
	}

	for i := 0; i < len(clients); i++ {
		if algnScore[i] >= THRESHOLD {
			trustScores[clients[i]] -= float32(DELTA)
		} else {
			trustScores[clients[i]] += float32(DELTA)
		}

		if trustScores[clients[i]] > 1 {
			trustScores[clients[i]] = 1.0
		} else if trustScores[clients[i]] < 0 {
			trustScores[clients[i]] = 0.0
		}
	}

	return trustScores, nil
}

func GetScorePath(modelID string, scoreType string) (*scpb.Score, error) {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "0.0.0.0:6379",
		Password: "",
		DB:       0,
	})

	key := fmt.Sprintf("%s!%s", modelID, scoreType)
	data, err := rdb.Get(ctx, key).Result()
	if err != nil {
		return &scpb.Score{}, err
	}

	err = rdb.Close()
	if err != nil {
		return &scpb.Score{}, err
	}

	score := &scpb.Score{}
	err = proto.Unmarshal([]byte(data), score)
	if err != nil {
		return &scpb.Score{}, err
	}

	return score, nil
}

func StoreScoreOnTangle(score scpb.Score) (string, error) {
	url := GOSHIMMER_NODE + "/formica"

	scoreBytes, err := proto.Marshal(&score)
	if err != nil {
		return "", err
	}

	payload := Block{
		Purpose: score.Type,
		Data:    scoreBytes,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(payloadBytes))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body)
	block := string(body)
	var response graph.Response
	json.Unmarshal(body, &response)
	if strings.Contains(block, "blockID") {
		return response.BlockID, nil
	}

	return "", errors.New(response.Error)
}

func isScore(purpose uint32) bool {
	scores_types := []int{SIMILARITY_PURPOSE_ID, PHI_PURPOSE_ID, GRADIENTS_PURPOSE_ID, TRUST_PURPOSE_ID, ALIGNMENT_PURPOSE_ID}
	for _, v := range scores_types {
		if uint32(v) == purpose {
			return true
		}
	}
	return false
}

func GetScoreByBlockID(blockID string) (scpb.Score, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	blockRaw, _ := goshimAPI.GetBlock(blockID)
	payload := new(formica.Payload)
	err := payload.FromBytes(blockRaw.Payload)
	if err != nil {
		return scpb.Score{}, err
	}

	if isScore(payload.Purpose()) {
		var score scpb.Score
		err := proto.Unmarshal([]byte(payload.Data()), &score)
		if err != nil {
			return scpb.Score{}, err
		}
		return score, nil
	}

	return scpb.Score{}, errors.New("Unknown payload type!")
}

func GetContentIPFS(path string) ([]byte, error) {
	url := IPFS_ENDPOINT + "/api/v0/get?arg=" + path
	req, err := http.NewRequest("POST", url, bytes.NewBuffer([]byte{}))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return []byte{}, err
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body)
	return body[512:len(body)], nil
}

func AddContentIPFS(content []byte) (string, error) {
	sh := shell.NewShell(IPFS_ENDPOINT)
	reader := bytes.NewReader(content)
	response, err := sh.Add(reader)
	if err != nil {
		return "", err
	}
	return response, nil
}

func StoreScoreLocally(modelID string, score scpb.Score) error {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "0.0.0.0:6379",
		Password: "",
		DB:       0,
	})

	scoreBytes, err := proto.Marshal(&score)
	if err != nil {
		return err
	}

	scoreType := ""
	if score.Type == TRUST_PURPOSE_ID {
		scoreType = "trust"
	} else if score.Type == SIMILARITY_PURPOSE_ID {
		scoreType = "similarity"
	} else if score.Type == PHI_PURPOSE_ID {
		scoreType = "phi"
	} else if score.Type == GRADIENTS_PURPOSE_ID {
		scoreType = "gradients"
	} else if score.Type == ALIGNMENT_PURPOSE_ID {
		scoreType = "algnscore"
	}

	key := fmt.Sprintf("%s!%s", modelID, scoreType)
	err = rdb.Set(ctx, key, scoreBytes, 0).Err()
	if err != nil {
		return err
	}

	err = rdb.Close()
	if err != nil {
		return err
	}

	return nil
}

func GetPhiFromNumpy(path string) ([]float64, error) {
	phiBytes, err := GetContentIPFS(path)
	if err != nil {
		return []float64{}, err
	}

	r, err := npyio.NewReader(bytes.NewReader(phiBytes))
	if err != nil {
		return []float64{}, err
	}

	var phi []float64
	err = r.Read(&phi)
	if err != nil {
		return []float64{}, err
	}

	return phi, nil
}

func GetAlignmentFromNumpy(path string) ([]float64, error) {
	algnBytes, err := GetContentIPFS(path)
	if err != nil {
		return []float64{}, err
	}

	r, err := npyio.NewReader(bytes.NewReader(algnBytes))
	if err != nil {
		return []float64{}, err
	}

	var algn []float64
	err = r.Read(&algn)
	if err != nil {
		return []float64{}, err
	}

	return algn, nil
}

func GetTrustFromNumpy(path string) ([]float64, error) {
	trustBytes, err := GetContentIPFS(path)
	if err != nil {
		return []float64{}, err
	}

	buf := bytes.NewBuffer(trustBytes)

	var trust []float64
	err = npyio.Read(buf, &trust)
	if err != nil {
		return []float64{}, err
	}

	return trust, nil
}

func GetSimilarityFromNumpy(path string) ([][]float64, error) {
	similarityBytes, err := GetContentIPFS(path)
	if err != nil {
		return [][]float64{}, err
	}

	r, err := npyio.NewReader(bytes.NewReader(similarityBytes))
	if err != nil {
		return [][]float64{}, err
	}

	var similarity [][]float64
	err = r.Read(&similarity)
	if err != nil {
		return [][]float64{}, err
	}

	return similarity, nil
}

func GetWeightsFromNumpy(path string) ([][]float64, error) {
	weightsBytes, err := GetContentIPFS(path)
	if err != nil {
		return [][]float64{}, err
	}

	buf := bytes.NewBuffer(weightsBytes)

	var weightsMat mat.Dense
	err = npyio.Read(buf, &weightsMat)
	if err != nil {
		fmt.Println("Error: Read", err.Error())
		return [][]float64{}, err
	}

	r, c := weightsMat.Dims()
	weights := make([][]float64, r)
	for i := range weights {
		weights[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			weights[i][j] = weightsMat.At(i, j)
		}
	}

	return weights, nil
}

func GetLatestGradients(modelID string) (map[string][][]float64, error) {
	var gradients Gradients
	scoreType := "gradients"
	latestGradient, err := GetScorePath(modelID, scoreType)
	if err != nil {
		return gradients.Content, err
	}

	latestGradientBytes, err := GetContentIPFS(latestGradient.Path)
	if err != nil {
		return gradients.Content, err
	}

	buf := bytes.NewBuffer(latestGradientBytes)
	dec := gob.NewDecoder(buf)
	err = dec.Decode(&gradients)
	if err != nil {
		return gradients.Content, err
	}

	return gradients.Content, nil
}

func GetLatestTrustScores(modelID string) (map[string]float32, error) {
	trustScores := make(map[string]float32)
	scoreType := "trust"
	score, err := GetScorePath(modelID, scoreType)
	if err != nil {
		return trustScores, err
	}

	latestTrustScores, err := GetTrustFromNumpy(score.Path)
	if err != nil {
		return trustScores, err
	}

	for i := 0; i < len(latestTrustScores); i++ {
		pubkey, err := graph.GetClientPubkey(i, modelID)
		if err != nil {
			return trustScores, err
		}
		trustScores[pubkey] = float32(latestTrustScores[i])
	}

	return trustScores, nil
}

func GetLatestRoundTimestamp(modelID string) (uint32, error) {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "0.0.0.0:6379",
		Password: "",
		DB:       0,
	})

	key := fmt.Sprintf("%s!timestamp", modelID)
	timestampStr, err := rdb.Get(ctx, key).Result()
	if err != nil {
		return uint32(time.Now().Unix()), err
	}

	err = rdb.Close()
	if err != nil {
		return uint32(time.Now().Unix()), err
	}

	timestamp, err := strconv.ParseUint(string(timestampStr), 10, 32)
	if err != nil {
		return uint32(time.Now().Unix()), err
	}

	return uint32(timestamp), nil
}

func StoreLatestRoundTimestamp(modelID string, timestamp uint32) error {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "0.0.0.0:6379",
		Password: "",
		DB:       0,
	})

	key := fmt.Sprintf("%s!timestamp", modelID)
	var timestampStr = strconv.FormatUint(uint64(timestamp), 10)

	err := rdb.Set(ctx, key, timestampStr, 0).Err()
	if err != nil {
		return err
	}

	err = rdb.Close()
	if err != nil {
		return err
	}

	return nil
}

func StoreGradientsIPFS(gradients map[string][][]float64) (string, error) {
	gradientsStruct := Gradients{Content: gradients}

	buf := new(bytes.Buffer)
	enc := gob.NewEncoder(buf)
	err := enc.Encode(gradientsStruct)
	if err != nil {
		return "", err
	}

	path, err := AddContentIPFS(buf.Bytes())
	if err != nil {
		return "", err
	}

	return path, nil
}

func substractMatrix(a [][]float64, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(a[0]))
	}

	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[0]); j++ {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result
}

func addMatrix(a [][]float64, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(a[0]))
	}

	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[i]); j++ {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

func GetLatestWeights(modelID string, clientPubkey string) ([][]float64, [][]float64, error) {
	latestTimstamp, err := GetLatestRoundTimestamp(modelID)
	if err != nil {
		return [][]float64{}, [][]float64{}, err
	}

	updates, err := graph.GetModelUpdates(modelID)
	if err != nil {
		return [][]float64{}, [][]float64{}, err
	}

	var updatesToProcess []*mupb.ModelUpdate
	for i := 0; i < len(updates); i++ {
		if updates[i].Timestamp > latestTimstamp && updates[i].Pubkey == clientPubkey {
			updatesToProcess = append(updatesToProcess, updates[i])
		}
	}

	if len(updatesToProcess) == 0 {
		return [][]float64{}, [][]float64{}, nil
	} else if len(updatesToProcess) < 2 {
		w1, err := GetWeightsFromNumpy(updatesToProcess[0].Weights)
		w2 := make([][]float64, len(w1))
		for i := range w2 {
			w2[i] = make([]float64, len(w1[0]))
		}
		if err != nil {
			return [][]float64{}, [][]float64{}, err
		}
		return w1, w2, nil
	} else {
		fst := rand.Intn(len(updatesToProcess))
		var scd int
		for {
			scd = rand.Intn(len(updatesToProcess))
			if scd != fst {
				break
			}
		}
		w1, err := GetWeightsFromNumpy(updatesToProcess[fst].Weights)
		if err != nil {
			return [][]float64{}, [][]float64{}, err
		}

		w2, err := GetWeightsFromNumpy(updatesToProcess[scd].Weights)
		if err != nil {
			return [][]float64{}, [][]float64{}, err
		}

		if fst > scd {
			return w1, w2, nil
		} else {
			return w2, w1, nil
		}
	}
}

func ComputeGradients(modelID string) (map[string][][]float64, error) {
	gradients := make(map[string][][]float64)
	clients, err := graph.GetClients(modelID)
	if err != nil {
		return gradients, err
	}

	// get the latest gradients for this client
	latestGradient, err := GetLatestGradients(modelID)
	if err != nil {
		fmt.Println("GetLatestGradients", err.Error())
		return gradients, err
	}

	for i := 0; i < len(clients); i++ {
		for j := 0; j < len(clients); j++ {
			if i == j {
				continue
			}
			// get randomly two weights of this client
			w1, w2, err := GetLatestWeights(modelID, clients[i])
			if err != nil {
				fmt.Println("GetLatestWeights", err.Error())
				return gradients, err
			}

			if len(w1) == 0 && len(w2) == 0 {
				continue
			}

			// substract the two weights
			substractWeights := substractMatrix(w1, w2)

			// add the substraction result to the latest gradients
			gradients[clients[i]] = addMatrix(latestGradient[clients[i]], substractWeights)
		}
	}

	// publish the new gradients
	gradientsPath, err := StoreGradientsIPFS(gradients)
	if err != nil {
		return gradients, err
	}

	score := scpb.Score{
		Type: GRADIENTS_PURPOSE_ID,
		Path: gradientsPath,
	}

	_, err = StoreScoreOnTangle(score)
	if err != nil {
		return gradients, err
	}

	err = StoreScoreLocally(modelID, score)
	if err != nil {
		return gradients, err
	}

	return gradients, nil
}

func ComputePhi(algnScores []float64) []float64 {
	phi := make([]float64, len(algnScores))
	max := float64(-1000.0)
	for i := 0; i < len(algnScores); i++ {
		phi[i] = 1.0 - algnScores[i]

		if phi[i] < 0 {
			phi[i] = 0
		}

		if phi[i] > 1 {
			phi[i] = 1
		}

		if phi[i] >= max {
			max = phi[i]
		}
	}

	for i := 0; i < len(algnScores); i++ {
		phi[i] = phi[i] / max
		if phi[i] == 1 {
			phi[i] = 0.99
		}
	}

	for i := 0; i < len(algnScores); i++ {
		phi[i] = math.Log(phi[i]/(1-phi[i])+0.000001) + 0.5
		if phi[i] > INF || phi[i] > 1 {
			phi[i] = 1.0
		}
		if phi[i] < 0 {
			phi[i] = 0.0
		}
	}

	// todo(lekssays): comment this
	for i := 0; i < len(algnScores); i++ {
		phi[i] = 1.0
	}
	return phi
}

func getClientIDLocally(modelID string, pubkey string) (int, error) {
	pubkeys, err := graph.GetClients(modelID)
	if err != nil {
		return -1, err
	}

	for i := 0; i < len(pubkeys); i++ {
		if pubkey == pubkeys[i] {
			return i, nil
		}
	}

	return -1, errors.New("id not found")
}

func converTrustToSlice(modelID string, trust map[string]float32) ([]float64, error) {
	trustScores := make([]float64, len(trust))
	for pubkey, score := range trust {
		id, err := getClientIDLocally(modelID, pubkey)
		if err != nil {
			return []float64{}, err
		}
		trustScores[id] = float64(score)
	}
	return trustScores, nil
}

func ConvertScoreToNumpy(modelID string, score interface{}, purpose uint32) (string, error) {
	buf := new(bytes.Buffer)
	if purpose == ALIGNMENT_PURPOSE_ID || purpose == PHI_PURPOSE_ID {
		content := score.([]float64)
		err := npyio.Write(buf, content)
		if err != nil {
			return "", err
		}
	} else if purpose == SIMILARITY_PURPOSE_ID {
		content := score.([][]float64)
		flattenedContent := flatten(content)
		m := mat.NewDense(len(content), len(content[0]), flattenedContent)
		err := npyio.Write(buf, m)
		if err != nil {
			return "", err
		}
	} else if purpose == TRUST_PURPOSE_ID {
		content := score.(map[string]float32)
		trustScores, err := converTrustToSlice(modelID, content)
		if err != nil {
			return "", err
		}

		err = npyio.Write(buf, trustScores)
		if err != nil {
			return "", err
		}
	} else {
		return "", errors.New("Undefined purpose!")
	}

	path, err := AddContentIPFS(buf.Bytes())
	if err != nil {
		return "", err
	}

	return path, err
}

func flatten(matrix [][]float64) []float64 {
	var result []float64

	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			result = append(result, matrix[i][j])
		}
	}

	return result
}

func PublishScore(modelID string, content interface{}, purpose uint32) error {
	if !isScore(purpose) {
		return errors.New("Undefined purpose.")
	}

	path, err := ConvertScoreToNumpy(modelID, content, purpose)
	if err != nil {
		return err
	}

	score := scpb.Score{
		Type: purpose,
		Path: path,
	}

	_, err = StoreScoreOnTangle(score)
	if err != nil {
		return err
	}

	err = StoreScoreLocally(modelID, score)
	if err != nil {
		return err
	}

	return nil
}

func Run(modelID string) error {
	_, err := ComputeGradients(modelID)
	if err != nil {
		return err
	}

	csMatrix, err := ComputeCSMatrix(modelID)
	if err != nil {
		return err
	}

	fmt.Println("csMatrix", csMatrix)
	algnScore := ComputeAlignmentScore(modelID, csMatrix)
	// csMatrix, algnScore, err = EvaluatePardoning(modelID, algnScore, csMatrix)
	// if err != nil {
	// 	return err
	// }

	err = PublishScore(modelID, csMatrix, SIMILARITY_PURPOSE_ID)
	if err != nil {
		return err
	}

	fmt.Println("algnScore", algnScore)
	err = PublishScore(modelID, algnScore, ALIGNMENT_PURPOSE_ID)
	if err != nil {
		fmt.Println("algnScore Error", err.Error())
		return err
	}

	phiScore := ComputePhi(algnScore)
	fmt.Println("Phi", phiScore)
	err = PublishScore(modelID, phiScore, PHI_PURPOSE_ID)
	if err != nil {
		return err
	}

	trustScore, err := ComputeTrust(modelID, algnScore)
	if err != nil {
		return err
	}

	fmt.Println("trustScore", trustScore)
	err = PublishScore(modelID, trustScore, TRUST_PURPOSE_ID)
	if err != nil {
		return err
	}

	currentTimestamp := uint32(time.Now().Unix())
	err = StoreLatestRoundTimestamp(modelID, currentTimestamp)
	if err != nil {
		return err
	}

	return nil
}

func Initialize(modelID string, x int, y int) error {
	currentTimestamp := uint32(time.Now().Unix())
	err := StoreLatestRoundTimestamp(modelID, currentTimestamp)
	if err != nil {
		return err
	}

	clients, err := graph.GetClients(modelID)
	if err != nil {
		return err
	}

	empty2DSlice := make([][]float64, x)
	for i := range empty2DSlice {
		empty2DSlice[i] = make([]float64, y)
	}

	empty1DSlice := make([]float64, len(clients))
	for i := 0; i < len(clients); i++ {
		empty1DSlice[i] = 0.0000

	}

	gradients := make(map[string][][]float64)
	for i := 0; i < len(clients); i++ {
		// initialize empty2DSlice with dimensions of the model (r and c)
		gradients[clients[i]] = empty2DSlice
	}

	gradientsPath, err := StoreGradientsIPFS(gradients)
	if err != nil {
		return err
	}

	score := scpb.Score{
		Type: GRADIENTS_PURPOSE_ID,
		Path: gradientsPath,
	}

	_, err = StoreScoreOnTangle(score)
	if err != nil {
		return err
	}

	err = StoreScoreLocally(modelID, score)
	if err != nil {
		return err
	}

	csMatrix := make([][]float64, len(clients))
	for i := range csMatrix {
		csMatrix[i] = make([]float64, len(clients))
	}

	fmt.Println("csMatrix", csMatrix)
	err = PublishScore(modelID, csMatrix, SIMILARITY_PURPOSE_ID)
	if err != nil {
		return err
	}

	err = PublishScore(modelID, empty1DSlice, ALIGNMENT_PURPOSE_ID)
	if err != nil {
		return err
	}

	phiScore := ComputePhi(empty1DSlice)
	fmt.Println("phiScore", phiScore)
	err = PublishScore(modelID, phiScore, PHI_PURPOSE_ID)
	if err != nil {
		return err
	}

	trustScore := make(map[string]float32)
	for i := 0; i < len(clients); i++ {
		trustScore[clients[i]] = float32(1.00)
	}

	fmt.Println("trustScore", trustScore)
	err = PublishScore(modelID, trustScore, TRUST_PURPOSE_ID)
	if err != nil {
		return err
	}

	return nil
}

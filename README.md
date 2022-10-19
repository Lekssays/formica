# :ant: formica
A DAG-based Fully-Decentralized Learning Framework


# Getting Started
## Dependencies
- Install golang v1.18

- Install python v3.8.10 and pip3

- Install docker v20.10.17 and docker-compose v2.6.1

- Install Redis

- Clone GoShimmer (IOTA 2.0) v0.9.8 in Home Directory: 

```
git clone https://github.com/iotaledger/goshimmer.git
```

- Clone formica in Home Directory: 

```
git clone https://github.com/lekssays/formica.git
```

## Install formica
- Copy formica plugin into Goshimmer folder:

```
cp -R $HOME/formica/protocol/plugins/formica $HOME/goshimmer/plugins/formica
```

- Integrate formica plugin in Goshimmer:
    - Add `"github.com/Lekssays/formica/protocol/plugins/formica"` to imports in `$HOME/goshimmer/plugins/research.go`
    - Add `formica.Plugin,"` after 	`chat.Plugin,` in `$HOME/goshimmer/plugins/research.go`
    - Install formica plugin locally 
    ```
    go get github.com/Lekssays/formica/protocol/plugins/formica
    ```

- Run GoShimmer Network: 

```
cd $HOME/goshimmer/tools/docker-network && ./run.sh
```

- Create formica network 

```
docker network create -d bridge formica
```

- Run IPFS node: 

```
docker run -d --name ipfs.formica.io --network="formica" -v $HOME/formica/simulator/ipfs/export:/export -v $HOME/formica/simulator/ipfs/data:/data/ipfs -p 4001:4001 -p 4001:4001/udp -p 0.0.0.0:8088:8088 -p 0.0.0.0:5001:5001 ipfs/go-ipfs:latest
```

- Install Python Dependecies: 

```
cd $HOME/formica/simulator/ && pip3 install -r requirements.txt
```

- Download and Unzip `data.zip` in `$HOME/formica/simulator/`

- Edit Environment Variables in `$HOME/formica/env.example`

- Rename `example.env` to `.env` and execute `source .env`

- Start Log Server: 

```
cd $HOME/formica/logs/ && python3 server.py
```

- Install golang dependencies 

```
cd $HOME/formica/protocol/ && go mod tidy && go build
```

- Start formica Host Listener

```
cd $HOME/formica/protocol/ && ./protocol listener
```

## Run formica

- To test if formica works, run:

```
cd $HOME/formica/simulator/ && python3 simulator.py -d MNIST
```

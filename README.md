# Serverless Simulator
- **Updated:** October 6, 2025

## Description

This simulator simulates the online process of application requests for data processing at remote/non-local devices. In theory, a request sends data (video, image, etc.) via HTTP from a user device to an application nested inside a container placed on a physical server at a datacenter somewhere. There, the data is processed, and the result is then returned back to the user.

![General Architecture](/docs/figures/general.PNG)

The simulator allows the following modifications:
- Network topology (you can simulate any network topology as long as its data follows /topology/edge.json)
- Network link bandwidth capacity (you can modify bandwidth of each network link)
- Application profile (you can put any application measurements in terms of CPU, RAM, data file size)
- Datacenter location and capacity (you can put any number of DCs at any network node, you can assign how many servers all/each DC should have)
- Request routing algorithm, request DCs forwarding algorithm (which DC should we forward the request to)
- Container placement algorithm (on which physical server in a specific DC should we place the container)
- Container warm time (the idle live time of serverless computing)
- Etc.

## Documentation

Please read documents in /docs/index.md before running.

## How to run

Install requirement packages:
```bash
pip install -r requirements.txt
```

Quick run, results will be printed at the end:
```bash
python3 main.py
```







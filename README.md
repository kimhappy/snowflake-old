<p align="center">
  <img width="750" src="logo.png" alt="Snowflake Logo"/>
</p>

<hr/>

**SNOWFLAKE** is a project using neural networks to simulate guitar amps/pedals. It draws inspiration from [GuitarML](https://github.com/GuitarML)'s [NeuralPi](https://github.com/GuitarML/NeuralPi) and [NeuralSeed](https://github.com/GuitarML/NeuralSeed) projects, and is compatible with their training code.

This project outperforms [Jatin Chowdhury](https://github.com/jatinchowdhury18)'s [RTNeural](https://github.com/jatinchowdhury18/RTNeural) in terms of inference speed for GRU and LSTM models. This performance boost allows the use of larger models in performance-constrained environments like [Raspberry Pi](https://www.raspberrypi.com) and [Daisy Seed](https://www.electro-smith.com/daisy/daisy). Also, with no external library dependencies, it is ideally suited for memory-constrained environments such as Daisy Seed.

This repository contains only the inference code and the CLI code for development. Once I establish a Raspberry Pi and Daisy Seed development environment, a new repository will be created for the complete project which can run on these devices.

Special thanks to GuitarML and Jatin Chowdhury for inspiring me to embark on this project.

## Build CLI
```sh
cd playground
./build.sh
```
* Use `chmod` to give build.sh the appropriate permissions, if necessary.
* For a clean build, use `./build.sh -f`

## Run CLI
```sh
./playground -i <input file name> -o <output file name> -b <samples per block>
```

## Create your own model
* Begin by using [Automated-GuitarAmpModelling](https://github.com/GuitarML/Automated-GuitarAmpModelling) to create your own .json model file.
* Then, use convert.py to transform this .json file into a .hpp file.
```sh
convert.py <json file name> <hpp file name> <amp name without whitespaces>
```
* At present, this project primarily targets running larger models on Daisy Seed, therefore, only a Python script is available to generate the .hpp file. Future plans include creating an audio plugin to run on a Raspberry Pi or desktop, which will necessitate the ability to convert to other formats.

## Benchmark Results
Please note: Benchmark results may vary depending on your environment.
I only measured the performance of the inference code.

* CPU: Apple M1 Pro
* Compiler: clang++ 16.0.1

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Library</th>
    <th>RTF</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://github.com/sdatkinson/NeuralAmpModelerPlugin">NAM</a></td>
    <td><a href="https://eigen.tuxfamily.org/index.php?title=Main_Page">Eigen</a></td>
    <td>0.1</td>
  </tr>
  <tr>
    <td><a href="https://github.com/GuitarML/SmartGuitarPedal">WaveNet</a></td>
    <td>Eigen</td>
    <td>0.09</td>
  </tr>
  <tr>
    <td rowspan="3"><a href="https://github.com/GuitarML/Proteus">LSTM 40</a></td>
    <td>RTNeural</td>
    <td>0.12</td>
  </tr>
  <tr>
    <td>RTNeural with Eigen</td>
    <td>0.022</td>
  </tr>
  <tr>
    <td>snowflake</td>
    <td>0.021</td>
  </tr>
  <tr>
    <td>LSTM 16</td>
    <td>snowflake</td>
    <td>0.0037</td>
  </tr>
  <tr>
    <td rowspan="3">GRU 10</td>
    <td>RTNeural</td>
    <td>0.0051</td>
  </tr>
  <tr>
    <td>RTNeural with Eigen</td>
    <td>0.0047</td>
  </tr>
  <tr>
    <td>snowflake</td>
    <td>0.0018</td>
  </tr>
  <tr>
    <td>GRU 20</td>
    <td>snowflake</td>
    <td>0.0046</td>
  </tr>
</tbody>
</table>

## TODO

* snowflake
  * Implement fade I/O for parameter change
  * Use open-source audio read/write library
  * Benchmark on the actual device and optimize the code

* snowflake-seed
  * Purchase a Daisy Seed
  * Purchase a PCB
    * PedalPCB [Terrarium](https://www.pedalpcb.com/product/pcb351)
      * Currently out of stock
    * [DaisySeedProjects](https://github.com/bkshepherd/DaisySeedProjects)
      * KiCad -> JLCPCB (with SMD) ???
    * I'm not familiar with this as my background is in CS XD
  * Build a pedal case
    * I aim to use a 3D printer for this, but lack the necessary expertise XD
  * Make a full project
    * Include more effects like Noise gate, Compressor, EQ, IR loader, Delay, Reverb, Chorus...
    * Develop an inverse filter for a cabinet solely for use with snowflake-seed

* snowflake-pi
  * Purchase a Raspberry Pi
  * Purchase a [Pisound](https://blokas.io/pisound)
  * Build a pedal case
  * Make a full project
    * Construct a signal chain (like Fractal Axe-FX) on the webpage
    * Include more effects like Noise gate, Compressor, EQ, IR loader, Delay, Reverb, Chorus...
    * Develop an inverse filter for a cabinet solely for use with snowflake-pi

---

This README.md was written by ChatGPT 🤖

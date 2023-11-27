# match-histogram-node
An InvokeAI node to match a histogram from one image to another.
- Option to only transfer luminance channels.
- Option to save output as grayscale

## Usage
Install: There are two options for installing these nodes. (Option 1 is the recommended option) 
1. Git clone the repo into the `invokeai/nodes` directory. (**Recommended** - as it allows updating via a git pull)
    - open a command prompt/terminal in the invokeAI nodes directory ( or choose `8. Open the developer console` option from the invoke.bat then `cd nodes`)
    - run `git clone https://github.com/skunkworxdark/match_histogram.git`
2. Manually download and place [match_histogram.py](match_histogram.py) & [__init__.py](__init__.py) in a sub folder in the `invokeai/nodes` folder.

<ins>Update:</ins><BR>
Run a `git pull` from the `XYGrid_nodes` folder. Or run the `update.bat` or `update.sh` that is in the `invokeai/nodes/match_histogram` folder. If you installed it manually then the only option is to monitor the repo or discord channel and manually download and replace the file yourself.

<ins>Remove:</ins><BR>
Simply delete the `XYGrid_nodes` folder or you can rename it by adding an underscore `_`XYGrid_nodes` and Invoke will ignore it.

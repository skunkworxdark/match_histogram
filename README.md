# match-histogram-node
An InvokeAI node to match a histogram from one image to another.
- Option to only transfer luminance channels.
- Option to save output as grayscale

## Usage
Install: There are two options for installing these nodes. (Option 1 is the recommended option) 
1. Git clone the repo into the `invokeai/nodes` directory. (**Recommended** - as it allows updating via a git pull)
    - open a command prompt/terminal in the invokeAI nodes directory ( or choose `8. Open the developer console` option from the invoke.bat then `cd nodes`)
    - run `git clone https://github.com/skunkworxdark/XYGrid_nodes.git`
2. Manually download and place [prompt_tools.py](prompt_tools.py) & [__init__.py](__init__.py) in a sub folder in the `invokeai/nodes` folder.

Important Note: If you have used a previous version of these nodes (Pre Invoke 3.4) that were installed in the .env invocations directory. Then the existing images_to_grids.py file must be deleted from the invocations directory otherwise invoke will throw an error with duplicate nodes. Also note that some of these nodes have changed names and parameters so existing workflows will need to be remade. See included workflows for examples.

<ins>Update:</ins><BR>
Run a `git pull` from the `XYGrid_nodes` folder. Or run the `update.bat` or `update.sh` that is in the `invokeai/nodes/XYGrid_nodes` folder. If you installed it manually then the only option is to monitor the repo or discord channel and manually download and replace the file yourself.

<ins>Remove:</ins><BR>
Simply delete the `XYGrid_nodes` folder or you can rename it by adding an underscore `_`XYGrid_nodes` and Invoke will ignore it.

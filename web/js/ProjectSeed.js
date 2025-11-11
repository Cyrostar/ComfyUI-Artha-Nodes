import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const nodeLastSeeds = new Map();

ComfyWidgets.BUTTON = function(node, inputName, inputData, app) {
    const widget = {
        type: "button",
        name: inputName,
        value: "",
        options: inputData[1] || {},

        draw: function(ctx, node, widget_width, y, widget_height) {
            const width_ratio = 10;
            const custom_height = 20;
            const margin = 8;

            const button_x = widget_width / width_ratio / 2;
            const button_y = y + margin / 2;
            const button_width = widget_width - (widget_width / width_ratio);
            const button_height = custom_height;

            // Draw button background
            ctx.fillStyle = "#4a4a6a";
            ctx.fillRect(button_x, button_y, button_width, button_height);
            
            // Draw button border
            ctx.strokeStyle = "#666688";
            ctx.lineWidth = 1;
            ctx.strokeRect(button_x, button_y, button_width, button_height);

            // Draw text
            ctx.fillStyle = "#ffffff";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            const text = this.options.text || "NONE";
            ctx.fillText(
                text,
                button_x + button_width / 2,
                button_y + button_height / 2
            );

            // Store bounds for hit detection
            this.last_y = y;
            this.last_h = custom_height + margin;
            this.last_bx = button_x;
            this.last_by = button_y;
            this.last_bw = button_width;
            this.last_bh = button_height;
        },

        mouse: function(event, pos, node) {
            if (event.type === "pointerup") {
                const inBounds = pos[0] >= this.last_bx && 
                               pos[0] <= this.last_bx + this.last_bw &&
                               pos[1] >= this.last_by && 
                               pos[1] <= this.last_by + this.last_bh;
                
                if (inBounds) {
                    if (this.options.callback) {
                        this.options.callback(node, this);
                    }
                    return true;
                }
            }
            return false;
        },
    };

    node.addCustomWidget(widget);
    return widget;
};

app.registerExtension({
    name: "ComfyUI.Artha.Nodes.Project.Seed",
    
    async setup(app) {
        // Track seeds when workflows are executed
        const originalQueuePrompt = app.queuePrompt;
        app.queuePrompt = async function(number, batchCount) {
            // Store current seeds before execution
            const projectSeedNodes = app.graph._nodes.filter(node => node.type === "Project Seed");
            
            for (const node of projectSeedNodes) {
                const seedWidget = node.widgets?.find(w => w.name === "seed");
                if (seedWidget && seedWidget.value) {
                    // Store this node's current seed as its "last executed seed"
                    nodeLastSeeds.set(node.id, seedWidget.value);
                    console.log(`Stored last seed for node ${node.id}: ${seedWidget.value}`);
                }
            }
            
            return originalQueuePrompt.call(this, number, batchCount);
        };
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        if (nodeData.name === "Artha Project Seed") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                
                onNodeCreated?.apply(this, arguments);
                
                this.size = [210, 130];
                this.minSize = [210, 130];
                this.setSize(this.size);
                
                this.color = "#222233";
                this.bgcolor = "#333355";
                
                const nodeRef = this;
                
                const buttonWidget = ComfyWidgets.BUTTON(this, "seed_button", [
                    "BUTTON", 
                    {
                        text: "USE LAST SEED",                        
                        callback: (node, widget) => {
                            
                            const seedWidget = nodeRef.widgets?.find(w => w.name === "seed");
                            
                            if (seedWidget) {
                                const lastSeed = nodeLastSeeds.get(nodeRef.id);
                                
                                if (lastSeed) {
                                    const oldValue = seedWidget.value;
                                    seedWidget.value = lastSeed;
                                    
                                    if (nodeRef.onWidgetChanged) {
                                        nodeRef.onWidgetChanged("seed", lastSeed, oldValue, seedWidget);
                                    }
                                    
                                    app.canvas.setDirty(true, true);
                            
                                }
						
                            }
                        }
                    }
                ]);
            };
        }
    }
});
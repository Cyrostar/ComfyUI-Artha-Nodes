import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Artha.Comfy.Image.Display",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
		if (nodeData.name === "Image Display") {
        
            const onNodeCreated = nodeType.prototype.onNodeCreated;
			
            nodeType.prototype.onNodeCreated = function () {
				
				onNodeCreated?.apply(this, arguments);
				
                // Create an <img> element for preview
				this.imageEl = document.createElement("img");
				this.imageEl.style.maxWidth = "100%";        // limit width to node
				this.imageEl.style.objectFit = "contain";    // preserve aspect ratio
				this.imageEl.style.border = "0px";
				this.imageEl.style.marginTop = "4px";
				this.imageEl.style.display = "none";
                this.addDOMWidget("Display", "image", this.imageEl);
				
            };
			
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            
			nodeType.prototype.onConnectionsChange = async function (type, slot, connected, link, ioSlot) {
				
				onConnectionsChange?.apply(this, arguments);
				
				if (connected && link) {
                    // Store the information about the node connected to our input
					this.connectedNodeId = link.origin_id;
                    this.connectedNodeSlot = link.origin_slot;
                } else {
                    // Clear the stored info if disconnected
                    this.connectedNodeId = null;
                    this.connectedNodeSlot = null;
                }
				
			};
			
			const onExecuted = nodeType.prototype.onExecuted;
			
            nodeType.prototype.onExecuted = async function (message) {
                
				onExecuted?.apply(this, arguments);
				
                if (!this.imageEl) return;

                // Free previous image
                this.imageEl.src = "";

                // Load new preview if exists
				if (message?.image && message.image.length > 0) {
					const base64 = message.image[0];
					this.imageEl.src = `data:image/png;base64,${base64}`;
					this.imageEl.style.display = "block";  // show
				} else {
					this.imageEl.src = "";
					this.imageEl.style.display = "none";  // hide
				}
				

			};                                        

        }
		
    }
	
});
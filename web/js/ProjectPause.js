import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Artha.PRJ.Project.Pause",
	
	setup() {
        // Listen for reset messages from Python
        api.addEventListener("artha_project_pause_button_reset", (event) => {
            const data = event.detail; 
            // Find all Project Pause nodes and reset their buttons
            const nodes = app.graph.findNodesByType("Project Pause");
            nodes.forEach(node => {
                if (node.button) {
					if(data.paused) {
						node.button.textContent = "RESUME";
						node.button.style.backgroundColor = "rgba(255, 218, 87, 0.7)";
					} else {
						node.button.textContent = "PAUSE";
						node.button.style.backgroundColor = "rgba(134, 165, 48, 0.7)";
					}
                }
            });
        });
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
		if (nodeData.name === "Project Pause") {
            
            if (!document.getElementById("artha_default_button_style")) {
                const style = document.createElement("style");
                style.id = "artha_default_button_style";
                style.textContent = `
                    .artha_default_button_style {
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        cursor: pointer;
                        border: 2px solid #000;
                        border-radius: 30px;
                        color: #000;
                        background: rgba(134, 165, 48, 0.7);
                        font-size: 12px;
                        font-weight: 400;
                        padding: 0.6em 1.2em;
                        transition: all .15s cubic-bezier(.4,0,.2,1);
                        width: 100%;
                        height: 30px !important;
                        min-height: 30px !important;
                        max-height: 30px !important;
                        flex-shrink: 0;
                        box-sizing: border-box;
                    }
                `;
				
                document.head.appendChild(style);
				
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                
				onNodeCreated?.apply(this, arguments);
				
				this.color = "#222233";
				this.bgcolor = "#333355";
                
                this.button = document.createElement("button");
                this.button.textContent = "PAUSE";
                this.button.classList.add("artha_default_button_style");
				
				this.button.onclick = () => {
										
					fetch('/artha/project_toggle_pause_button', {
						method: 'POST',
						headers: {
							'Content-Type': 'application/json',
						},
						body: JSON.stringify({
							'action': 'toggle',
							'node_id': this.id
						})
					})
					.then(response => response.json())
					.then(data => {
						// Update button text to reflect state
						if(data.paused) {
							this.button.textContent = "RESUME";
							this.button.style.backgroundColor = "rgba(255, 218, 87, 0.7)";
						} else {
							this.button.textContent = "PAUSE";
							this.button.style.backgroundColor = "rgba(134, 165, 48, 0.7)";
						}
					})
					.catch(error => {
						console.error('Error:', error);
					});					
					
				};

                const arthaWidgetButton = this.addDOMWidget("Pause Button", "button", this.button);
				
				if (arthaWidgetButton) {
					
                    arthaWidgetButton.serialize = true;
                }
								
            };
        }
    }
});
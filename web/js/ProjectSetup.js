import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Artha.PRJ.Project.Setup",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
		if (nodeData.name === "Artha Project Setup") {
        
            const onNodeCreated = nodeType.prototype.onNodeCreated;
			
            nodeType.prototype.onNodeCreated = function () {
				
				onNodeCreated?.apply(this, arguments);
				
				this.color = "#222233";
				this.bgcolor = "#333355";

            }; 			

        }
		
    }
	
});
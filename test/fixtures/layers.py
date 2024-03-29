from libs.model.layertemplate import LayerTemplate
from libs.model_helpers import activators

relu_4_9 = LayerTemplate('relu', 4, 9, activators.relu, activators.drelu)
relu_4_4 = LayerTemplate('relu', 4, 4, activators.relu, activators.drelu)
relu_1_4 = LayerTemplate('relu', 1, 4, activators.relu, activators.drelu)
relu_1_9 = LayerTemplate('relu', 1, 9, activators.relu, activators.drelu)
relu_1_1 = LayerTemplate('relu', 1, 1, activators.relu, activators.drelu)
relu_1_3 = LayerTemplate('relu', 1, 3, activators.relu, activators.drelu)
relu_3_1 = LayerTemplate('relu', 3, 1, activators.relu, activators.drelu)

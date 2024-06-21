# DermAItool's Flask sub-server

This server is mainly responsible for the current models deployment, and anything related to predictions or GRAD-CAM generation. Its routing is only accessible via HTTP requests from the SpringBoot's dedicated REST API, thus the "sub" prefixe.

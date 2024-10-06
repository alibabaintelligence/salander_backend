@echo off
echo This batch runs all the commands necessary to update the Azure container with FastAPI code (only in this project)

REM Command 1
echo Updating local content of the container
docker build -t salanderregistry.azurecr.io/salander-backend:latest .

REM Command 2
echo Push the new updated local changes
docker push salanderregistry.azurecr.io/salander-backend:latest

REM Command 3
echo Update web app and make it use the new container
az webapp config container set --name SalanderBackend --resource-group salander-group --docker-custom-image-name salanderregistry.azurecr.io/salander-backend:latest

REM Command 4
echo Restart web app to make sure it picks up changes
az webapp restart --name SalanderBackend --resource-group salander-group

echo All update commands have been executed
pause
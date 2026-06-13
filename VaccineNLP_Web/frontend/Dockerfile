FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY --chown=node:node . .
RUN chown -R node:node /app/node_modules
USER node
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://127.0.0.1:5173/ || exit 1
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]

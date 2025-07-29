FROM node:18-alpine

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all source files
COPY . .

# Create .env file if it doesn't exist
RUN echo "REACT_APP_API_URL=http://localhost:8000" > .env

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"] 
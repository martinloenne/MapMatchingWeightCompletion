FROM adoptopenjdk/openjdk11:jdk-11.0.8_10-alpine


# Add dependencies
# Build application
# Run application
COPY . .

# Get maven package
RUN apk add maven
RUN chmod +x mvnw
RUN mvn package
#RUN mvn clean test

# NB! This will NOT publish the port but is only for documentation purposes. 
EXPOSE 5000

ENTRYPOINT ["java","-jar","/target/mapmatchingservice-1.0.0-beta.jar"]

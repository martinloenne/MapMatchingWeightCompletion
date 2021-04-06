package com.mapmatching.integration;

import com.mapmatching.web.Application;
//import com.mapmatching.web.models.GpxDTO;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;


@SpringBootTest(classes = Application.class)
@AutoConfigureMockMvc
public class MapMatchControllerIntegration {
    @Autowired
    private MockMvc mockMvc;
    
    @Test
    public void PostRoot_statusIsOk() throws Exception {        
        mockMvc.perform(post("/")).andExpect(status().isOk());
    }

    @Test
    public void PostMapMatch_ReceivedData_PostTripValidData_statusIsOk() throws Exception {
        String json = Files.readString(Path.of("files/trip_valid.json"));
         mockMvc.perform(post("/mapmatch")
                 .contentType(MediaType.APPLICATION_JSON)
                 .content(json))
                 .andExpect(status().isOk());
    }
    @Test
    public void PostMapMatch_ReceivedData_PostTripDataOutsideMap_statusIsBadRequest() throws Exception {
        String json = Files.readString(Path.of("files/trip_outside_map.json"));
        mockMvc.perform(post("/mapmatch")
                 .contentType(MediaType.APPLICATION_JSON)
                 .content(json))
                 .andExpect(status().isBadRequest());
    }
    @Test
    public void PostMapMatch_ReceivedData_PostTripDataWithoutPoints_statusIsBadRequest() throws Exception {
        String json = Files.readString(Path.of("files/trip_no_points.json"));
        mockMvc.perform(post("/mapmatch")
                .contentType(MediaType.APPLICATION_JSON)
                .content(json))
                .andExpect(status().isBadRequest());
    }
    @Test
    public void PostMapMatch_ReceivedData_PostTripDataInvalidLatitude_statusIsBadRequest() throws Exception {
        String json = Files.readString(Path.of("files/trip_invalid_latitude.json"));
        mockMvc.perform(post("/mapmatch")
                .contentType(MediaType.APPLICATION_JSON)
                .content(json))
                .andExpect(status().isBadRequest());
    }
    @Test
    public void PostMapMatch_ReceivedData_PostTripDataInvalidLongitude_statusIsBadRequest() throws Exception {
        String json = Files.readString(Path.of("files/trip_invalid_longitude.json"));
        mockMvc.perform(post("/mapmatch")
                .contentType(MediaType.APPLICATION_JSON)
                .content(json))
                .andExpect(status().isBadRequest());
    }

    @Test
    public void PostMapMatch_ReceivedData_PostTripDataInvalidJsonData_statusIsBadRequest() throws Exception {
        String json = Files.readString(Path.of("files/trip_invalid_json.json"));
        mockMvc.perform(post("/mapmatch")
                .contentType(MediaType.APPLICATION_JSON)
                .content(json))
                .andExpect(status().isBadRequest());
    }
}
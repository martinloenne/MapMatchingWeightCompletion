package com.mapmatching.unit;

import com.mapmatching.core.helpers.EdgeFunctions;
import com.mapmatching.core.models.input.*;
import com.mapmatching.core.models.output.*;
import com.mapmatching.unit.helpers.MyAssert;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class EdgeFunctionsTests {

    private RawPoint point;
    private Edge nonEmptyEdge1;
    private Edge nonEmptyEdge2;
    private Edge nonEmptyEdge3;
    private Edge nonEmptyEdge4;
    private Edge emptyEdge1;
    private Edge emptyEdge2;
    private Edge emptyEdge3;


    @BeforeEach
    public void setUp() {

        // ---------- For getEmptyEdges tests -------------- //
        point = new RawPoint(0, 0, null);

        nonEmptyEdge1 = new Edge();
        nonEmptyEdge1.getSnappedPoints().add(point);
        nonEmptyEdge2 = new Edge();
        nonEmptyEdge2.getSnappedPoints().add(point);
        nonEmptyEdge3 = new Edge();
        nonEmptyEdge3.getSnappedPoints().add(point);
        nonEmptyEdge4 = new Edge();
        nonEmptyEdge4.getSnappedPoints().add(point);

        emptyEdge1 = new Edge();
        emptyEdge2 = new Edge();
        emptyEdge3 = new Edge();
    }

    @Test
    public void getEmptyEdges_CallWithEmptyEdgesAtIndex_ReturnsCorrectEmptyEdges() {
        // Arrange
        List<Edge> edges = new ArrayList<>();
        edges.add(nonEmptyEdge1);
        edges.add(emptyEdge1);
        edges.add(emptyEdge2);
        edges.add(nonEmptyEdge2);

        List<Edge> expectedResult = new ArrayList<>();
        expectedResult.add(emptyEdge1);
        expectedResult.add(emptyEdge2);

        // Act
        List<Edge> actualResult = EdgeFunctions.getSequenceEmptyEdges(edges, 1);

        // Assert
        MyAssert.equals(expectedResult, actualResult);
    }

    @Test
    public void getEmptyEdges_CallWithEmptyEdgesAfterIndex_ReturnsNoEmptyEdges() {
        // Arrange
        List<Edge> edges = new ArrayList<>();
        edges.add(nonEmptyEdge1);
        edges.add(nonEmptyEdge2);
        edges.add(emptyEdge1);
        edges.add(emptyEdge2);

        // Act
        List<Edge> result = EdgeFunctions.getSequenceEmptyEdges(edges, 1);

        // Assert
        MyAssert.isTrue(result.size() == 0);
    }

    @Test
    public void getEmptyEdges_CallWithEmptyEdgesAtIndexAndLaterIndex_ReturnsCorrectEmptyEdges() {
        // Arrange
        List<Edge> edges = new ArrayList<>();
        edges.add(nonEmptyEdge1);
        edges.add(emptyEdge1);
        edges.add(emptyEdge2);
        edges.add(nonEmptyEdge2);
        edges.add(emptyEdge3);

        List<Edge> expectedResult = new ArrayList<>();
        expectedResult.add(emptyEdge1);
        expectedResult.add(emptyEdge2);

        // Act
        List<Edge> actualResult = EdgeFunctions.getSequenceEmptyEdges(edges, 1);

        // Assert
        MyAssert.equals(expectedResult, actualResult);
    }

    @Test
    public void calcDistanceBetweenEmptyEdges_CallWithStartEdgeEndEdgeAndEmptyEdges_ReturnsCorrectDistance() {
        Edge startingEdge = new Edge();
        RawPoint startingEdgeSnappedPoint = new RawPoint(30.6335756, 104.0549158, null);
        startingEdge.getSnappedPoints().add(startingEdgeSnappedPoint);

        Edge endingEdge = new Edge();
        RawPoint endingEdgeSnappedPoint = new RawPoint(30.6334478, 104.0533089, null);
        endingEdge.getSnappedPoints().add(endingEdgeSnappedPoint);

        Node emptyEdge1StartNode = new Node(30.633552, 104.0546572);
        Node emptyEdge2EndNode = new Node(30.6334478, 104.0535125);
        emptyEdge1.setStartNode(emptyEdge1StartNode);
        emptyEdge2.setEndNode(emptyEdge2EndNode);

        emptyEdge1.setDistance(100);
        emptyEdge2.setDistance(50);

        List<Edge> emptyEdges = new ArrayList<>();
        emptyEdges.add(emptyEdge1);
        emptyEdges.add(emptyEdge2);

        double expectedResult = 194;

        // Act
        double actualResult = EdgeFunctions.calcDistanceBetweenEmptyEdges(startingEdge, endingEdge, emptyEdges);

        // Assert
        MyAssert.isWithinThreshold(expectedResult, actualResult, 0.5); // 0.5 meter error margin
    }

    @Test
    public void calcSpeedBetweenEmptyEdges_CallWithStartEdgeEndEdgeAndEmptyEdges_ReturnsCorrectSpeed() {
        // Arrange
        Date startTime = new Date(2020, 10, 25, 12, 0, 0);
        Date endTime = new Date(2020, 10, 25, 12, 0, 20);

        Edge startingEdge = new Edge();
        RawPoint startingEdgeSnappedPoint = new RawPoint(30.6335756, 104.0549158, startTime);
        startingEdge.getSnappedPoints().add(startingEdgeSnappedPoint);

        Edge endingEdge = new Edge();
        RawPoint endingEdgeSnappedPoint = new RawPoint(30.6334478, 104.0533089, endTime);
        endingEdge.getSnappedPoints().add(endingEdgeSnappedPoint);

        Node emptyEdge1StartNode = new Node(30.633552, 104.0546572);
        Node emptyEdge2EndNode = new Node(30.6334478, 104.0535125);
        emptyEdge1.setStartNode(emptyEdge1StartNode);
        emptyEdge2.setEndNode(emptyEdge2EndNode);

        emptyEdge1.setDistance(100);
        emptyEdge2.setDistance(50);

        List<Edge> emptyEdges = new ArrayList<>();
        emptyEdges.add(emptyEdge1);
        emptyEdges.add(emptyEdge2);

        double expectedResult = 9.7;

        // Act
        double actualResult = EdgeFunctions.calcSpeedBetweenEmptyEdges(startingEdge, endingEdge, emptyEdges);

        // Assert
        MyAssert.isWithinThreshold(expectedResult, actualResult, 0.1); // 0.1 m/s error margin
    }

    @Test
    public void calcSpeedFromEdgePoints_CallWithEdgeContainingMultiplePoints_ReturnsCorrectSpeed() throws Exception {
        // Arrange
        Date timePoint1 = new Date(2020, 10, 25, 12, 0, 0);
        Date timePoint2 = new Date(2020, 10, 25, 12, 0, 5);
        Date timePoint3 = new Date(2020, 10, 25, 12, 0, 30);

        RawPoint point1 = new RawPoint(30.6335946, 104.055136, timePoint1);
        RawPoint point2 = new RawPoint(30.6335567, 104.0547507, timePoint2);
        RawPoint point3 = new RawPoint(30.6335378, 104.0543325, timePoint3);

        Edge edge = new Edge();
        edge.getSnappedPoints().add(point1);
        edge.getSnappedPoints().add(point2);
        edge.getSnappedPoints().add(point3);

        double expectedResult = 2.5;

        // Act
        double actualResult = EdgeFunctions.calcSpeedFromEdgePoints(edge);

        // Assert
        MyAssert.isWithinThreshold(expectedResult, actualResult, 0.2);
    }

    @Test
    public void calcSpeedFromSingleEdgePoint_CallWithTwoEdgesContainingSinglePoints_ReturnsCorrectSpeed() {
        // Arrange
        Date timePoint1 = new Date(2020, 10, 25, 12, 0, 0);
        Date timePoint2 = new Date(2020, 10, 25, 12, 0, 10);

        RawPoint point1 = new RawPoint(30.632702, 104.0547948, timePoint1);
        RawPoint point2 = new RawPoint(30.632818, 104.0544756, timePoint2);

        Node edge1EndNode = new Node(30.632818, 104.0544756);
        Node edge2StartNode = new Node(30.632818, 104.0544756);

        Edge edge1 = new Edge();
        edge1.setEndNode(edge1EndNode);
        edge1.getSnappedPoints().add(point1);

        Edge edge2 = new Edge();
        edge2.setStartNode(edge2StartNode);
        edge2.getSnappedPoints().add(point2);

        List<Edge> edges = new ArrayList<>();
        edges.add(edge1);
        edges.add(edge2);

        double expectedResult = 3.4;

        // Act
        double actualResult = EdgeFunctions.calcSpeedFromSingleEdgePoint(edge1, edges);

        // Assert
        MyAssert.isWithinThreshold(expectedResult, actualResult, 0.1);
    }

    @Test
    public void calcSpeedBetweenOverlappingEdges_CallWithTwoEdgesAndOverlappingNode_ReturnsCorrectSpeed() {
        // Arrange
        Date firstPointTime = new Date(2020, 10, 25, 12, 0, 0);
        Date secondPointTime = new Date(2020, 10, 25, 12, 0, 30);

        RawPoint firstEdgePoint = new RawPoint(30.6335946, 104.055136, firstPointTime);
        RawPoint secondEdgePoint = new RawPoint(30.6335378, 104.0543325, secondPointTime);

        Date endNodeTime = new Date(2020, 10, 25, 12, 0, 5);
        Node firstEdgeEndNode = new Node(30.6335567, 104.0547507);
        firstEdgeEndNode.setTimestamp(endNodeTime);

        Edge firstEdge = new Edge();
        firstEdge.getSnappedPoints().add(firstEdgePoint);
        firstEdge.setEndNode(firstEdgeEndNode);

        Edge secondEdge = new Edge();
        secondEdge.getSnappedPoints().add(secondEdgePoint);

        double expectedResult = 2.5;

        // Act
        double actualResult = EdgeFunctions.calcSpeedBetweenOverlappingEdges(firstEdge, secondEdge, firstEdge.getEndNode());

        // Assert
        MyAssert.isWithinThreshold(expectedResult, actualResult, 0.2);
    }
}

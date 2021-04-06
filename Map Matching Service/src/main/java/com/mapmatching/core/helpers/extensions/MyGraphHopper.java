/*
 *  Licensed to GraphHopper and Peter Karich under one or more contributor
 *  license agreements. See the NOTICE file distributed with this work for
 *  additional information regarding copyright ownership.
 *
 *  GraphHopper licenses this file to you under the Apache License,
 *  Version 2.0 (the "License"); you may not use this file except in
 *  compliance with the License. You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package com.mapmatching.core.helpers.extensions;

import com.graphhopper.reader.DataReader;
import com.graphhopper.reader.ReaderNode;
import com.graphhopper.reader.osm.GraphHopperOSM;
import com.graphhopper.storage.DataAccess;
import com.graphhopper.storage.Directory;
import com.graphhopper.storage.GraphHopperStorage;
import com.graphhopper.util.BitUtil;

/**
 *
 * @author Peter Karich
 */
public class MyGraphHopper extends GraphHopperOSM {

  // mapping of internal edge ID to OSM way ID
  private DataAccess edgeMapping;
  // mapping of internal node ID to OSM node ID
  private DataAccess towerNodeMapping;
  private DataAccess pillarNodeMapping;
  private BitUtil bitUtil;

  @Override
  public boolean load(String graphHopperFolder) {
    boolean loaded = super.load(graphHopperFolder);

    Directory dir = getGraphHopperStorage().getDirectory();
    bitUtil = BitUtil.get(dir.getByteOrder());

    towerNodeMapping = dir.find("tower_node_mapping");
    pillarNodeMapping = dir.find("pillar_node_mapping");
    edgeMapping = dir.find("edge_mapping");

    if (loaded) {
      towerNodeMapping.loadExisting();
      pillarNodeMapping.loadExisting();
      edgeMapping.loadExisting();
    }
    // XXX: Else??

    return loaded;
  }

  @Override
  protected DataReader createReader(GraphHopperStorage ghStorage) {

    MyOSMReader reader = new MyOSMReader(ghStorage) {
      {
        towerNodeMapping.create(2000);
        pillarNodeMapping.create(2000);
        edgeMapping.create(1000);
      }

      // this method is only in >0.6 protected, before it was private
      @Override
      protected void storeOsmWayID(int edgeId, long osmWayId) {

        super.storeOsmWayID(edgeId, osmWayId);

        long pointer = 8L * edgeId;
        edgeMapping.ensureCapacity(pointer + 8L);

        edgeMapping.setInt(pointer, bitUtil.getIntLow(osmWayId));
        edgeMapping.setInt(pointer + 4, bitUtil.getIntHigh(osmWayId));
      }

      @Override
      protected boolean addNode(ReaderNode node) {
        boolean result = super.addNode(node);

        if (result) {
          int internalNodeId = this.getNodeMap().get(node.getId());
          storeOsmNodeID(internalNodeId, node.getId());
        }
        return result;
      }

      protected void storeOsmNodeID(int nodeId, long osmNodeId) {
        final DataAccess nodeMapping;
        if (nodeId < 0) {
          // if nodeId < 0 then this is a tower node
          nodeId = -nodeId;
          nodeMapping = towerNodeMapping;
        } else {
          // if nodeId > 0 then this is a pillar node
          nodeMapping = pillarNodeMapping;
        }
        // Not sure why the node process adds 3 to the node id?
        // Possibly as tower and pillar node are internally stored in the same map,
        // The +3 removes the conflict where id == 0, which would result in tower == -0,
        // pillar == 0
        nodeId -= 3;
        long pointer = 8L * nodeId;
        nodeMapping.ensureCapacity(pointer + 8L);
        nodeMapping.setInt(pointer, bitUtil.getIntLow(osmNodeId));
        nodeMapping.setInt(pointer + 4, bitUtil.getIntHigh(osmNodeId));
      }

      @Override
      protected void finishedReading() {
        super.finishedReading();

        towerNodeMapping.flush();
        pillarNodeMapping.flush();
        edgeMapping.flush();
      }

    };

    return initDataReader(reader);
  }

  public long getOSMWay(int internalEdgeId) {
    long pointer = 8L * internalEdgeId;
    return bitUtil.combineIntsToLong(edgeMapping.getInt(pointer), edgeMapping.getInt(pointer + 4L));
  }

  public long getTowerOSMNode(int internalNodeId) {
    return getOSMNode(internalNodeId, towerNodeMapping);
  }

  public long getPillarOSMNode(int internalNodeId) {
    return getOSMNode(internalNodeId, pillarNodeMapping);
  }

  public long getOSMNode(int internalNodeId, DataAccess nodeMapping) {
    try {
      long pointer = 8L * internalNodeId;
      return bitUtil.combineIntsToLong(nodeMapping.getInt(pointer), nodeMapping.getInt(pointer + 4L));
    } catch (ArrayIndexOutOfBoundsException e) {
      System.out.println("Node id (" + internalNodeId + ") out of bounds");
      return -1;
    }
  }
}
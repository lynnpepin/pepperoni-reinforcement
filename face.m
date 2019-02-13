classdef face < handle
   properties
      index
      vertex1=circle_vertex.empty
      vertex2=circle_vertex.empty
      vertex3=circle_vertex.empty
      halfedge1=halfedge.empty
      halfedge2=halfedge.empty
      halfedge3=halfedge.empty
   end
   methods
       function obj=face(v1,v2,v3,i)
           if nargin > 0
               obj.index=i;
               obj.vertex1=v1;
               obj.vertex2=v2;
               obj.vertex3=v3;
               obj.halfedge1=halfedge(v1,v2,i);
               obj.halfedge2=halfedge(v2,v3,i);
               obj.halfedge3=halfedge(v3,v1,i);

               obj.halfedge1.Next=obj.halfedge2;
               obj.halfedge2.Next=obj.halfedge3;
               obj.halfedge3.Next=obj.halfedge1;
               if isempty(obj.vertex1.incident_halfedge)
                   obj.vertex1.incident_halfedge=obj.halfedge1;
               end
               if isempty(obj.vertex2.incident_halfedge)
                   obj.vertex2.incident_halfedge=obj.halfedge2;
               end
               if isempty(obj.vertex3.incident_halfedge)
                   obj.vertex3.incident_halfedge=obj.halfedge3;
               end
           end
       end
   end
end
classdef halfedge < handle
%
%
   properties
       source = circle_vertex.empty
       target = circle_vertex.empty
       face_index
       Flip 
       Next = halfedge.empty
       Prev = halfedge.empty
   end
   
   methods
      function obj=halfedge(vi,vj,f)
         % Construct a dlnode object
         if nargin > 0
            obj.source=vi;
            obj.target=vj;
            obj.face_index=f;
         end
      end
     
   end
   
end

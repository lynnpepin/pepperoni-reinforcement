classdef circle_vertex < handle
    properties
        index
        radius
        x
        y
        neighbors=circle_vertex.empty(1,0)
        incident_halfedge
        placed=0
        totall_angle
    end
    methods
        function obj=circle_vertex(i,r,ang)
            if nargin > 0
                obj.index=i;
                obj.radius=r;
                obj.totall_angle=ang;
            end
        end
    end

end
%Circle Packing
function Circles = CirclePacking(T,R) % T: the triangulation connectivity relationship, R: the radius and surround angle
n_tri=length(T);
n_circles=length(R);

Circles = circle_vertex.empty(n_circles,0); % Output of the function, arrary of circle_vertex
for i=1:n_circles
    Circles(i)=circle_vertex(i,R(i,1),R(i,2)); % generate circle_vertex with circle index, radius, surround angle
end

Faces = face.empty(n_tri,0); % the faces of the triangulation. face ~ triangle
i_he=1;
for i=1:n_tri
    Faces(i)=face(Circles(T(i,1)),Circles(T(i,2)),Circles(T(i,3)),i); 
    Halfedges(i_he)=Faces(i).halfedge1; % storing halfedges of every face
    Halfedges(i_he+1)=Faces(i).halfedge2;
    Halfedges(i_he+2)=Faces(i).halfedge3;
    i_he=i_he+3;

end

for j=1:length(Halfedges) % find the flip of every halfedge
    for i=1:length(Halfedges)
       if Halfedges(i).source.index == Halfedges(j).target.index &&...
               Halfedges(i).target.index == Halfedges(j).source.index
           Halfedges(j).Flip=Halfedges(i);

       end
       
    end
end



%find distinguish boundary and interior circles
j=1;
k=1;
for i=1:n_circles
    if Circles(i).totall_angle == 2*pi
        Ci(j) = Circles(i);
        j=j+1;
    else
        Cb(k) = Circles(i);
        k=k+1;
    end
   
end
% find boundary circles in ccw sequence 
% for i=1:length(Halfedges)
%     if isempty(Halfedges(i).Flip)
%         traveller=Halfedges(i);
%         Cb(1)=traveller.source;
%         Cb(2)=traveller.target;
%     end
% end
% while Cb(end).index ~= Cb(1).index
%     while ~isempty(traveller.Next.Flip)
%         traveller=traveller.Next.Flip;
%     end
%     traveller=traveller.Next;
%     Cb(end+1)=traveller.target;
% end
% 
% %find interior circles
% k=1;
% for i=1:n_circles
%     if ~in_circles(Circles(i),Cb)
%         Ci(k)=Circles(i);
%         k=k+1;
%     end
% end

% find neighbor for every interior circles
for i=1:length(Ci)
    traveller2 = Ci(i).incident_halfedge;
    Ci(i).neighbors(1) = traveller2.target;
    traveller2 = traveller2.Next.Next.Flip;
    Ci(i).neighbors(2) = traveller2.target;
    while Ci(i).neighbors(end).index ~= Ci(i).neighbors(1).index
        traveller2 = traveller2.Next.Next.Flip;
        Ci(i).neighbors(end+1)=traveller2.target;
    end
    
end

% find neighbor for every boundary circles
for i=1:length(Cb)
   traveller3 = Cb(i).incident_halfedge;
   if isempty(traveller3.Flip)
       Cb(i).neighbors(1) = traveller3.target;
       Cb(i).neighbors(2) = traveller3.Next.target;
       while ~isempty(traveller3.Next.Next.Flip)
           traveller3 = traveller3.Next.Next.Flip;
           Cb(i).neighbors(end+1) = traveller3.Next.target;
       end
   else
       while ~isempty(traveller3.Flip)
           traveller3 = traveller3.Flip.Next;
       end
       Cb(i).neighbors(1) = traveller3.target;
       Cb(i).neighbors(2) = traveller3.Next.target;
       while ~isempty(traveller3.Next.Next.Flip)
           traveller3 = traveller3.Next.Next.Flip;
           Cb(i).neighbors(end+1) = traveller3.Next.target;
       end
   end
       
end

% calculate interior radii
% eps = 0.01;
% delta_r = 0.001;
% theta_diff = ones(length(Ci),1);
% for i=1:length(Ci)
%    theta_diff(i) = theta_arround(Ci(i)) - 2*pi; 
% end
% while max(abs(theta_diff)) > eps
%    for i=1:length(Ci)
%       if theta_diff(i) < 0
%           Ci(i).radius = Ci(i).radius - delta_r*Ci(i).radius;
%           theta_diff(i) = theta_arround(Ci(i)) - 2*pi; 
%       elseif theta_diff(i) > 0
%           Ci(i).radius = Ci(i).radius + delta_r*Ci(i).radius;
%           theta_diff(i) = theta_arround(Ci(i)) - 2*pi; 
%       end
%    end
% end

% Calculate radii for interior and undertermined circles
eps = 0.01;
delta_r = 0.001;
theta_diff = zeros(n_circles,1);
for i=1:n_circles
    if Circles(i).totall_angle ~= 0
        theta_diff(i) = theta_arround(Circles(i)) - Circles(i).totall_angle; 
    else
        theta_diff(i) = 0;
    end
end
while max(abs(theta_diff)) > eps
   for i=1:n_circles
      if theta_diff(i) < 0
          Circles(i).radius = Circles(i).radius - delta_r*Circles(i).radius;
          theta_diff(i) = theta_arround(Circles(i)) - Circles(i).totall_angle; 
      elseif theta_diff(i) > 0
          Circles(i).radius = Circles(i).radius + delta_r*Circles(i).radius;
          theta_diff(i) = theta_arround(Circles(i)) - Circles(i).totall_angle; 
      end
   end
end

% Layout circles
Faces_copy = face.empty(n_tri,0);
for i=1:length(Faces)
   Faces_copy(i) = Faces(i); 
end

% ancher
Circles(19).x=0;
Circles(19).y=0;
Circles(19).placed=1;
Circles(9).x=0;
Circles(9).y=Circles(19).y+Circles(19).radius+Circles(9).radius;
Circles(9).placed=1;

i_face=1;
while ~isempty(Faces_copy)
    if Faces_copy(i_face).vertex1.placed+...
            Faces_copy(i_face).vertex2.placed+Faces_copy(i_face).vertex3.placed == 3
        Faces_copy(i_face) = [];
        i_face = 1;
    elseif  Faces_copy(i_face).vertex1.placed+...
            Faces_copy(i_face).vertex2.placed+Faces_copy(i_face).vertex3.placed == 2
        if Faces_copy(i_face).vertex1.placed && Faces_copy(i_face).vertex2.placed
            vi = Faces_copy(i_face).vertex1;
            vj = Faces_copy(i_face).vertex2;
            vk = Faces_copy(i_face).vertex3;
        elseif Faces_copy(i_face).vertex2.placed && Faces_copy(i_face).vertex3.placed
            vi = Faces_copy(i_face).vertex2;
            vj = Faces_copy(i_face).vertex3;
            vk = Faces_copy(i_face).vertex1;
        else
            vi = Faces_copy(i_face).vertex3;
            vj = Faces_copy(i_face).vertex1;
            vk = Faces_copy(i_face).vertex2;
        end
        theta_ij = atan2(vj.y-vi.y, vj.x-vi.x );
        ri = vi.radius;
        rj = vj.radius;
        rk = vk.radius;
        alpha_i = acos( ((ri+rj)^2+(ri+rk)^2-(rj+rk)^2)/(2*(ri+rj)*(ri+rk)) );
        vk.x = vi.x + (ri+rk)*cos(alpha_i+theta_ij);
        vk.y = vi.y + (ri+rk)*sin(alpha_i+theta_ij);
        vk.placed = 1;
        Faces_copy(i_face)=[];
        i_face = 1;
        
    else
        i_face= i_face + 1;
        if i_face > length(Faces_copy)
            i_face = 1;
        end
    end
    
end

% plot Circle Packing
figure
hold on;
for i=1:length(Circles)
    viscircles([Circles(i).x Circles(i).y],Circles(i).radius);
end

end
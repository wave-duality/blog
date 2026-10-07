"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export default function Antiprism() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

    renderer.setSize(320, 320);
    renderer.setClearColor(0xffffff, 0);
    container.appendChild(renderer.domElement);

    const vertices = [
      [-1, -0.6, -1],
      [1, -0.6, -1],
      [1, -0.6, 1],
      [-1, -0.6, 1],
      [0, 0.6, -1.414],
      [1.414, 0.6, 0],
      [0, 0.6, 1.414],
      [-1.414, 0.6, 0],
    ];

    const faces = [
      { indices: [0, 1, 2, 3], color: 0x4a9eff },
      { indices: [4, 5, 6, 7], color: 0xff6b6b },
      { indices: [0, 1, 4], color: 0x95e1d3 },
      { indices: [1, 2, 5], color: 0x95e1d3 },
      { indices: [2, 3, 6], color: 0x95e1d3 },
      { indices: [3, 0, 7], color: 0x95e1d3 },
      { indices: [0, 4, 7], color: 0xf38181 },
      { indices: [1, 4, 5], color: 0xf38181 },
      { indices: [2, 5, 6], color: 0xf38181 },
      { indices: [3, 6, 7], color: 0xf38181 },
    ];

    const geometry = new THREE.BufferGeometry();
    const positions: number[] = [];
    const colors: number[] = [];
    const idxs: number[] = [];
    let vertexIndex = 0;

    faces.forEach((face) => {
      const startIndex = vertexIndex;
      face.indices.forEach((idx) => {
        const v = vertices[idx];
        positions.push(v[0], v[1], v[2]);
        colors.push(
          (face.color >> 16) / 255,
          ((face.color >> 8) & 255) / 255,
          (face.color & 255) / 255
        );
        vertexIndex++;
      });

      if (face.indices.length === 4) {
        idxs.push(
          startIndex, startIndex + 1, startIndex + 2,
          startIndex, startIndex + 2, startIndex + 3
        );
      } else if (face.indices.length === 3) {
        idxs.push(startIndex, startIndex + 1, startIndex + 2);
      }
    });

    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(idxs);
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      flatShading: true,
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const edges = new THREE.EdgesGeometry(geometry);
    const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 1.5 });
    const wireframe = new THREE.LineSegments(edges, edgeMaterial);
    scene.add(wireframe);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    camera.position.z = 4;

    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    const rotation = { x: 0.3, y: 0.3 };

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        rotation.y += (e.clientX - previousMousePosition.x) * 0.01;
        rotation.x += (e.clientY - previousMousePosition.y) * 0.01;
        previousMousePosition = { x: e.clientX, y: e.clientY };
      }
    };

    const onMouseUp = () => { isDragging = false; };

    container.addEventListener("mousedown", onMouseDown);
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);

    let animationId: number;
    function animate() {
      animationId = requestAnimationFrame(animate);
      mesh.rotation.x = rotation.x;
      mesh.rotation.y = rotation.y;
      wireframe.rotation.x = rotation.x;
      wireframe.rotation.y = rotation.y;
      renderer.render(scene, camera);
    }
    animate();

    return () => {
      cancelAnimationFrame(animationId);
      container.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      edgeMaterial.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return <div ref={containerRef} className="w-[320px] h-[320px] cursor-grab active:cursor-grabbing" />;
}

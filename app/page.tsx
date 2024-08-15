"use client"

import {NavigationMenuDemo} from "@/components/section/navbar";
import { AvatarDemo } from "@/components/section/avatar";
import { ButtonDemo } from "@/components/section/button";
import { InputDemo } from "@/components/section/input";
import { BadgeDemo } from "@/components/section/badge";
import { Textarea } from "@/components/ui/textarea";
import { Table } from "@/components/ui/table";
//import { MyChart } from "@/components/section/chart";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <NavigationMenuDemo/>
      <AvatarDemo/>
      <ButtonDemo/>
      <InputDemo/>
      <BadgeDemo/>
      <Textarea/>
      <Table/>
    </main>
  );
}